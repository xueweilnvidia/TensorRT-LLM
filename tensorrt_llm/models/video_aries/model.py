from dataclasses import dataclass
from typing import Optional, Union
from collections import OrderedDict
import tensorrt as trt

from ..._utils import str_dtype_to_trt, trt_dtype_to_str, int32_array, int64_array
from ...functional import (Tensor, allgather, arange, chunk, dynamic_chunk, concat, constant,
                           cos, exp, expand, shape, silu, sin, slice, split, dynamic_split,
                           unsqueeze, identity, cast)
from ...functional import sum as trtllm_sum_func
from ...layers import MLP, BertAttention, Conv2d, Embedding, LayerNorm, Linear
from ...lora_manager import (LoraConfig, use_lora)
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...parameter import Parameter
from ...plugin import current_all_reduce_helper
from ...quantization import QuantMode
from ..modeling_utils import PretrainedConfig, PretrainedModel
from .config import AriesParamsConfig
from .aries_layers import (DoubleStreamBlock, SingleStreamBlock, LastLayer,
                           SingleTokenRefiner,MLPEmbedder, timestep_embedding,
                           PatchEmbed, TimestepEmbedder,  TextProjection)



       
class AriesModel(Module):
    """
    Transformer model for flow matching on sequences.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.
    """
    def check_config(self, config: AriesParamsConfig):
        config.set_if_not_exist('input_h', 192)
        config.set_if_not_exist('input_w', 336)
        config.set_if_not_exist('frames', 129)
        config.set_if_not_exist('in_channels', 16)
        config.set_if_not_exist('out_channels', 16)
        config.set_if_not_exist('vec_in_dim', 768)
        config.set_if_not_exist('context_in_dim', 4096)
        config.set_if_not_exist('hidden_size', 3072)
        config.set_if_not_exist('mlp_ratio', 4.0)
        config.set_if_not_exist('num_heads', 24)
        config.set_if_not_exist('depth_double_blocks', 20)
        config.set_if_not_exist('depth_single_blocks', 40)
        config.set_if_not_exist('theta', 10000)
        config.set_if_not_exist('rope_dim_list', [16,56,56])
        config.set_if_not_exist('qkv_bias', True)
        config.set_if_not_exist('guidance_embed', False)
        #config.set_if_not_exist('attn_mode', "ootb")
        config.set_if_not_exist('attn_mode', "plugin")

    def __init__(
        self,
        config: AriesParamsConfig,
    ):
        #factory_kwargs = {'device': device, 'dtype': dtype}
        self.check_config(config)
        super().__init__()
        
        self.config = config
        self.mapping = config.mapping
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.dtype = str_dtype_to_trt(config.dtype)
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.unpatchify_channels = self.out_channels
        self.rope_dim_list = config.rope_dim_list
        self.attn_mode = config.attn_mode #ootb, plugin
        print(f"********AriesModel attn_mode={self.attn_mode}")
        self.guidance_embed = config.guidance_embed

        #self.mlp_act_type = 'gelu_tanh'
        self.mlp_act_type = 'gelu_pytorch_tanh'
        self.patch_size = [1,2,2]
        self.use_vanilla = False
        self.reverse = True
        self.qk_norm = True,
        self.qk_norm_type = 'rms'


        factory_kwargs = { 'dtype': self.dtype}
        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.text_projection = 'single_refiner'
        self.text_states_dim = config.context_in_dim
        self.use_attention_mask = True

        # Text pooling. The pooled vector will be added to the timestep as a global context.
        # See more details (Lumina-Next): http://arxiv.org/abs/2406.18583
        #                  (Flux.1): https://github.com/black-forest-labs/flux
        self.text_pool_type = 'clip'
        self.text_states_dim_2 = config.vec_in_dim

        # Now we only use above configs from config.
        # TODO(ckczzjzhang): align Flux configs with DiT.

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"Hidden size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
            )
        pe_dim = self.hidden_size // self.num_heads
        if sum(self.rope_dim_list) != pe_dim:
            raise ValueError(f"Got {self.rope_dim_list} but expected positional dim {pe_dim}")

        if self.use_vanilla:
            print("use_vanilla unsupported!")
            assert False
            self.img_in = Linear(
                self.in_channels * self.patch_size * self.patch_size,
                self.hidden_size,
                bias=True,
                dtype = self.dtype
            )
            self.txt_in = Linear(self.text_states_dim, self.hidden_size, **factory_kwargs)
            self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, **factory_kwargs)
            self.vector_in = MLPEmbedder(self.text_states_dim_2, self.hidden_size, **factory_kwargs)
            self.guidance_in = MLPEmbedder(
                in_dim=256,
                hidden_dim=self.hidden_size,
                **factory_kwargs
            ) if self.guidance_embed else None
        else:
            # image
            self.img_in = PatchEmbed(
                self.patch_size,
                self.in_channels,
                self.hidden_size,
                **factory_kwargs
            )

            # text
            if self.text_projection == "linear":
                self.txt_in = TextProjection(
                    self.text_states_dim,
                    self.hidden_size,
                    "silu",
                    **factory_kwargs
                )
            elif self.text_projection == "single_refiner":
                self.txt_in = SingleTokenRefiner(
                    self.text_states_dim,
                    self.hidden_size,
                    self.num_heads,
                    depth=2,
                    mapping=self.mapping,
                    **factory_kwargs
                )
            else:
                raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

            # time modulation
            self.time_in = TimestepEmbedder(
                self.hidden_size,
                "silu",
                **factory_kwargs
            )

            # text modulation
            self.vector_in = MLPEmbedder(
                self.text_states_dim_2,
                self.hidden_size,
                **factory_kwargs
            ) if self.text_pool_type is not None else None

            # guidance modulation
            self.guidance_in = TimestepEmbedder(
                self.hidden_size,
                "silu",
                **factory_kwargs
            ) if self.guidance_embed else None

        # blocks
        self.double_blocks = ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    mlp_act_type=self.mlp_act_type,
                    qk_norm=self.qk_norm,
                    qk_norm_type=self.qk_norm_type,
                    qkv_bias=config.qkv_bias,
                    attn_mode=self.attn_mode,
                    reverse=self.reverse,
                    mapping=self.mapping,
                    **factory_kwargs
                )
                for _ in range(config.depth_double_blocks)
            ]
        )

        self.single_blocks = ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    mlp_act_type=self.mlp_act_type,
                    qk_norm=self.qk_norm,
                    qk_norm_type=self.qk_norm_type,
                    attn_mode=self.attn_mode,
                    reverse=self.reverse,
                    mapping=self.mapping,
                    **factory_kwargs
                )
                for _ in range(config.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            mapping=self.mapping,
            #get_activation_layer("silu"),
            **factory_kwargs
        )

    def set_attn_mode(self, new_mode):
        for block in self.double_blocks:
            block.set_attn_mode(new_mode)
        for block in self.single_blocks:
            block.set_attn_mode(new_mode)

    def forward(
        self,
        x: Tensor,
        t: Tensor, # Should be in range(0, 1000).
        text_states: Tensor = None,
        text_mask: Tensor = None,
        text_states_2: Optional[Tensor] = None, # Text embedding for modulation.
        freqs_cos: Optional[Tensor] = None,
        freqs_sin: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        guidance: Tensor = None, # Guidance for modulation, should be cfg_scale x 1000.
    ) -> Tensor:
        img = x
        txt = text_states
        ot = x.shape[2]
        oh = shape(x,3)
        ow = shape(x,4)
        tt, th, tw = ot // self.patch_size[0], oh // self.patch_size[1], ow // self.patch_size[2]
        # Prepare modulation vectors.
        # time modulation
        if self.use_vanilla:
            t = timestep_embedding(t, 256).cast(img.dtype)
            vec = self.time_in(t)
            print("use_vanilla unsupported!")
        else:
            # our timestep_embedding is merged into time_in(TimestepEmbedder)
            vec = self.time_in(t)

        # text modulation
        if self.use_vanilla:
            #vec = vec + self.vector_in(text_states_2)
            print("use_vanilla unsupported!")
        else:
            if self.text_pool_type is not None:
                if self.text_pool_type == "clip":
                    vec = vec + self.vector_in(text_states_2)
                else:
                    raise NotImplementedError(f"Unsupported text_pool_type: {self.text_pool_type}")
            else:
                pass

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            if self.use_vanilla:
                guidance = timestep_embedding(guidance, 256).cast(img.dtype)
                vec = vec + self.guidance_in(guidance)
            else:
                # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
                vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        if self.use_vanilla:
            #img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=self.patch_size, pw=self.patch_size)
            #img = self.img_in(img)
            #txt = self.txt_in(txt)
            print("use_vanilla unsupported!")
        else:
            img = self.img_in(img)
            if self.text_projection == "linear":
                txt = self.txt_in(txt)
            elif self.text_projection == "single_refiner":
                txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
            else:
                raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        attn_input_len, attn_max_input_len = None, None
        if self.mapping.cp_size > 1:
            #assert img.shape[1] % self.mapping.cp_size == 0 
            img = dynamic_chunk(img, self.mapping.cp_size, dim=1)[self.mapping.cp_rank]
            txt = chunk(txt, self.mapping.cp_size, dim=1)[self.mapping.cp_rank]
            text_mask = chunk(text_mask, self.mapping.cp_size, dim=1)[self.mapping.cp_rank]
            if freqs_cos is not None:
                freqs_cos = dynamic_chunk(freqs_cos,  self.mapping.cp_size, dim=0)[self.mapping.cp_rank]
                freqs_sin = dynamic_chunk(freqs_sin,  self.mapping.cp_size, dim=0)[self.mapping.cp_rank]
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        txt_seq_len = shape(txt,1)
        img_seq_len = shape(img,1)
        if self.attn_mode == "plugin":
            valid_txt_len = trtllm_sum_func(text_mask, dim=1)
            self.register_network_output("valid_txt_len", valid_txt_len)
            attn_input_len = trtllm_sum_func(text_mask, dim=1)+img_seq_len.cast(trt.DataType.INT32)
            attn_input_len = cast(attn_input_len, trt.DataType.INT32)
            #attn_input_len.mark_output("output_attn_input_len", trt.DataType.INT32)
            self.register_network_output("img_seq_len", img_seq_len)
            self.register_network_output("attn_input_len", attn_input_len)
            #attn_max_input_len = shape(txt,1) + shape(img,1)
            #attn_max_input_len = cast(attn_max_input_len, trt.DataType.INT32)

        # --------------------- Pass through Flux blocks ------------------------
        for layer_num, block in enumerate(self.double_blocks):
            double_block_args = [img, txt, attn_mask, vec, attn_input_len, attn_max_input_len, freqs_cis]
            img, txt = block(*double_block_args)

        # Merge txt and img to pass through single stream blocks.
        if self.reverse:
            x = concat([img, txt], dim=1)
        else:
            x = concat([txt, img], 1)

        # Compatible with MMDiT.
        self.register_network_output("single_block_input_x", x)
        if len(self.single_blocks) > 0:
            for layer_num, block in enumerate(self.single_blocks):
                single_block_args = [x, attn_mask, vec, txt.shape[1], attn_input_len, attn_max_input_len, (freqs_cos, freqs_sin)]
                x = block(*single_block_args)
        
        if self.reverse:
            #img = x[:, :img_seq_len, ...]
            start_index = img_seq_len
            img, img_after_split = dynamic_split(x, [start_index, shape(x,1)-start_index], dim=1)
        else:
            #img = x[:, txt_seq_len:, ...]
            start_index = txt_seq_len
            img_before_split, img = dynamic_split(x, [start_index, shape(x,1)-start_index], dim=1)

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        self.register_network_output("final_layer", img)
        # All gather after CP
        if self.mapping.cp_size > 1:
            img = allgather(img, self.mapping.cp_group, gather_dim=1)

        img = self.unpatchify(img, tt, th, tw)
        return img

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        #assert t * h * w == x.shape[1]

        x = x.view(shape=concat([x.shape[0], t, h, w, c, pt, ph, pw]))
        #x = torch.einsum('nthwcopq->nctohpwq', x)
        x = x.permute([0, 4, 1, 5, 2, 6, 3, 7])
        imgs = x.view(shape=concat([x.shape[0], c, t * pt, h * ph, w * pw]))

        return imgs
 



class AriesModelForLM(PretrainedModel):
    def __init__(self, config: AriesParamsConfig):
        super().__init__(config)
        self.model = AriesModel(config)
        self.quant_mode = config.quant_mode
        self.mapping = config.mapping
        self.input_h = config.input_h
        self.input_w = config.input_w
        self.frames = config.frames
        self.dtype = str_dtype_to_trt(config.dtype)

    #def __post_init__(self):
    #    return
    def prepare_inputs(self, max_batch_size, **kwargs):
        '''@brief: Prepare inputs Tensors for the model, the given sizes are used to determine the ranges of the dimensions of when using TRT dynamic shapes.
           @return: a list contains values which can be fed into the self.forward()
        '''
        mapping = self.config.mapping
        def aries_default_range(max_batch_size):
            return [1, max(1, (max_batch_size + 1) // 2), max_batch_size]
        default_range = aries_default_range
        min_latent_h = 720 // 8
        min_latent_w = 720 // 8
        latent_h = self.input_h // 8
        latent_w = self.input_w // 8
        max_latent_h = 1280 // 8
        max_latent_w = 1280 // 8
        latent_frames = (self.frames - 1)//4 + 1
        min_img_seq_len = min_latent_h * min_latent_w //4*latent_frames
        img_seq_len = latent_h * latent_w //4*latent_frames
        max_img_seq_len = max_latent_h * max_latent_w //4*latent_frames
        txt_len = 256
        seq_len = img_seq_len + txt_len
        img = Tensor(
                name='x', dtype=self.dtype,
                shape=[-1, self.config.in_channels, latent_frames, -1, -1],
                dim_range=OrderedDict([
                    ('batch_size', [default_range(max_batch_size)]),
                    ('in_channels',[[self.config.in_channels]*3]),
                    ('frame_nums',[[latent_frames]*3]),
                    ('latent_h',[[min_latent_h, latent_h, max_latent_h]]),
                    ('latent_w',[[min_latent_w, latent_w, max_latent_w]]),
              ]))
        timestep = Tensor(
                name='t', dtype=trt.float32,
                shape=[-1],
                dim_range=OrderedDict([
                    ('batch_size', [default_range(max_batch_size)]),
              ]))
        txt = Tensor(
                name='text_states', dtype=self.dtype,
                shape=[-1, txt_len, 4096],
                dim_range=OrderedDict([
                    ('batch_size', [default_range(max_batch_size)]),
                    ('txt_len',[[txt_len]*3]),
                    ('text_states_dim',[[self.config.context_in_dim]*3]),
              ]))

        text_mask = Tensor(
                name='text_mask', dtype=trt.int32,
                shape=[-1, txt_len],
                dim_range=OrderedDict([
                    ('batch_size', [default_range(max_batch_size)]),
                    ('txt_len',[[txt_len]*3]),
              ]))
        prompt_embeds_2 = Tensor(
                name='text_states_2', dtype=self.dtype,
                shape=[-1, self.config.vec_in_dim],
                dim_range=OrderedDict([
                    ('batch_size', [default_range(max_batch_size)]),
                    ('vec_in_dim',[[self.config.vec_in_dim]*3]),
              ]))
        freqs_cos = Tensor(
                name='freqs_cos', dtype=trt.float32,
                shape=[-1, self.config.head_size],
                dim_range=OrderedDict([
                    ('img_seq_len',[[min_img_seq_len, img_seq_len, max_img_seq_len]]),
                    ('head_size',[[self.config.head_size]*3]),
              ]))
        freqs_sin = Tensor(
                name='freqs_sin', dtype=trt.float32,
                shape=[-1, 128],
                dim_range=OrderedDict([
                    ('img_seq_len',[[min_img_seq_len, img_seq_len, max_img_seq_len]]),
                    ('head_size',[[self.config.head_size]*3]),
              ]))

        model_inputs_map= {'x':img, 'text_states':txt, 'text_mask':text_mask, 'text_states_2':prompt_embeds_2,
                't':timestep, 'freqs_cos':freqs_cos, 'freqs_sin':freqs_sin}

        if self.config.attn_mode == "ootb":
            attn_num_heads = self.config.num_heads//mapping.tp_size
            cp_seq_len = seq_len // mapping.cp_size
            cp_txt_len = txt_len // mapping.cp_size
            cp_min_img_seq_len = min_img_seq_len // mapping.cp_size
            cp_max_img_seq_len = max_img_seq_len // mapping.cp_size
            attn_mask = Tensor(
                    name='attn_mask', dtype=self.dtype,
                    shape=[-1, self.config.num_heads//mapping.tp_size, -1, -1],
                    dim_range=OrderedDict([
                        ('batch_size', [default_range(max_batch_size)]),
                        ('dim1',[[attn_num_heads]*3]),
                        ('atn_seq_len_dim2',[[cp_min_img_seq_len + cp_txt_len, cp_seq_len, cp_max_img_seq_len + cp_txt_len]]),
                        ('atn_seq_len_dim3',[[min_img_seq_len + txt_len, seq_len, max_img_seq_len + txt_len]]),
                  ]))
            model_inputs_map["attn_mask"]=attn_mask

        if self.config.guidance_embed:
            guidance = Tensor(
                    name='guidance', dtype=self.dtype,
                    shape=[-1],
                    dim_range=OrderedDict([
                        ('batch_size', [default_range(max_batch_size)]),
                  ]))
            model_inputs_map["guidance"]=guidance
        print(f"********AriesModel model_inputs_name={model_inputs_map.keys()}")
        return model_inputs_map

    def forward(self,
                x: Tensor,
                t: Tensor,
                text_states: Tensor,
                text_mask: Tensor,
                text_states_2: Tensor,
                freqs_cos: Tensor,
                freqs_sin: Tensor,
                attn_mask: Optional[Tensor] = None,
                guidance: Optional[Tensor] = None):

        kwargs = {
            'x': x,
            't': t,
            'text_states': text_states,
            'text_mask': text_mask,
            'text_states_2': text_states_2,
            'freqs_cos': freqs_cos,
            'freqs_sin': freqs_sin,
            'attn_mask': attn_mask,
            'guidance': guidance,
        }
        output = self.model.forward(**kwargs)
        output.mark_output('noise_pred', self.dtype)
        return output

    def default_plugin_config(self, **kwargs):
        plugin_config = super().default_plugin_config(**kwargs)
        if self.quant_mode.is_int4_weight_only_per_group():
            plugin_config.weight_only_groupwise_quant_matmul_plugin = 'auto'
        return plugin_config

    def use_lora(self, lora_config: LoraConfig):
        use_lora(self, lora_config, self.trtllm_modules_to_torch_modules)
