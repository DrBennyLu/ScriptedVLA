"""
基于Flow Matching的动作头
用于从视觉-语言特征预测动作轨迹
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from torch.distributions import Beta


def swish(x):
    """Swish激活函数"""
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        self.d_model = d_model
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [B, T] 或 [B, T, 1] 时间步索引（整数）
        Returns:
            [B, T, d_model] 位置编码
        """
        if x.dim() == 2:
            # [B, T] -> [B, T, d_model]
            # 注意：这里使用 x 中的值作为索引，而不是直接使用 x.size(1)
            # 但为了简化，我们使用序列长度来索引位置编码
            B, T = x.shape
            # 扩展 batch 维度：[1, T, d_model] -> [B, T, d_model]
            pe_output = self.pe[:, :T, :].expand(B, -1, -1)
            return pe_output
        elif x.dim() == 3:
            # [B, T, 1] -> [B, T, d_model]
            B, T, _ = x.shape
            pe_output = self.pe[:, :T, :].expand(B, -1, -1)
            return pe_output
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")


class MLP(nn.Module):
    """简单的MLP"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    """动作编码器，将动作和时间步编码为特征"""
    
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)
    
    def forward(self, actions, timesteps):
        """
        Args:
            actions: [B, T, action_dim] 动作序列
            timesteps: [B] 每个batch的时间步标量
        Returns:
            [B, T, hidden_size] 编码后的动作特征
        """
        B, T, _ = actions.shape
        
        # 扩展时间步到所有T步
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)  # [B] -> [B, T]
        else:
            raise ValueError(
                f"Expected `timesteps` to have shape (B,), got {timesteps.shape}"
            )
        
        # 动作编码
        a_emb = self.layer1(actions)  # [B, T, hidden_size]
        
        # 时间步编码
        # pos_encoding 期望输入是 [B, T]，返回 [B, T, hidden_size]
        # timesteps 已经是 [B, T] 形状（从上面的扩展得到）
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)  # [B, T, hidden_size]
        
        # 拼接并处理
        x = torch.cat([a_emb, tau_emb], dim=-1)  # [B, T, 2*hidden_size]
        x = swish(self.layer2(x))  # [B, T, hidden_size]
        x = self.layer3(x)  # [B, T, hidden_size]
        
        return x


class DiTBlock(nn.Module):
    """
    DiT基础块（用于Flow Matching的Transformer）
    支持交叉注意力机制
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_cross_attention: bool = True
    ):
        super().__init__()
        
        self.use_cross_attention = use_cross_attention
        
        # 自注意力
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 交叉注意力（如果使用）
        if use_cross_attention:
            self.norm_cross = nn.LayerNorm(hidden_dim)
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        # MLP
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, hidden_dim] 输入序列
            encoder_hidden_states: [B, encoder_seq_len, hidden_dim] VLM特征（用于交叉注意力）
            encoder_attention_mask: [B, encoder_seq_len] 编码器注意力掩码
            attn_mask: 自注意力掩码
        """
        # 自注意力
        residual = x
        x = self.norm1(x)
        x_attn, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = residual + x_attn
        
        # 交叉注意力（如果使用）
        if self.use_cross_attention and encoder_hidden_states is not None:
            residual = x
            x = self.norm_cross(x)
            x_cross, _ = self.cross_attn(
                x,
                encoder_hidden_states,
                encoder_hidden_states,
                key_padding_mask=encoder_attention_mask
            )
            x = residual + x_cross
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        
        return x


class FlowMatchingActionHead(nn.Module):
    """
    基于Flow Matching的动作预测头
    从视觉-语言特征预测机器人动作轨迹
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        action_dim: int = 7,
        action_horizon: int = 10,  # 动作序列长度
        dropout: float = 0.1,
        use_cross_attention: bool = True,
        state_dim: Optional[int] = None,
        num_target_vision_tokens: int = 32,
        max_seq_len: int = 1024,
        add_pos_embed: bool = True,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        num_timestep_buckets: int = 1000,
        num_inference_timesteps: int = 50,
        cross_attention_dim: Optional[int] = None
    ):
        """
        初始化Flow Matching动作头
        
        Args:
            hidden_dim: Transformer隐藏层维度
            num_layers: Transformer层数
            num_heads: 注意力头数
            mlp_ratio: MLP扩展比例
            action_dim: 动作维度
            action_horizon: 动作序列长度（未来动作窗口大小）
            dropout: Dropout比例
            use_cross_attention: 是否使用交叉注意力
            state_dim: 状态维度（如果使用状态）
            num_target_vision_tokens: 目标视觉token数量（future_tokens的数量）
                - 作用：可学习的占位符token，通过交叉注意力从VLM特征中提取信息
                - 如何确定：通常通过实验调优（16, 32, 64等），或参考VLM输出的视觉token数量
                - 默认值：32（经验值，在大多数任务中表现良好）
            max_seq_len: 最大序列长度
            add_pos_embed: 是否添加位置编码
            noise_beta_alpha: Beta分布alpha参数
            noise_beta_beta: Beta分布beta参数
            noise_s: Flow matching噪声参数
            num_timestep_buckets: 时间步离散化桶数
            num_inference_timesteps: 推理时的去噪步数
            cross_attention_dim: VLM hidden_states的维度（用于交叉注意力，如果与hidden_dim不同则需要投影）
        """
        super().__init__()
        
        # 确保 training 属性被初始化（nn.Module 会自动设置，但显式初始化更安全）
        self.training = True
        
        self.hidden_dim = hidden_dim
        self.cross_attention_dim = cross_attention_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.use_cross_attention = use_cross_attention
        self.add_pos_embed = add_pos_embed
        self.num_timestep_buckets = num_timestep_buckets
        self.num_inference_timesteps = num_inference_timesteps
        
        # 状态编码器（如果使用状态）
        if state_dim is not None and state_dim > 0:
            self.state_encoder = MLP(
                input_dim=state_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim
            )
        else:
            self.state_encoder = None
        
        # 动作编码器
        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            hidden_size=hidden_dim
        )
        
        # Future tokens（用于动作预测的占位符）
        # 作用：
        # 1. 提供可学习的占位符token，用于从VLM的视觉-语言特征中提取信息
        # 2. 通过交叉注意力机制，让DiT块能够关注VLM的hidden_states
        # 3. 这些token在训练过程中学习如何将视觉-语言信息转换为动作预测的上下文
        # 4. 类似于Transformer中的特殊token（如[CLS]），但专门用于动作预测任务
        # 
        # num_target_vision_tokens: 占位符token的数量（超参数，通常设置为32）
        # 这个值可以通过以下方式确定：
        # - 实验调优：尝试不同的值（16, 32, 64等），选择性能最好的
        # - 参考VLM输出：可以设置为VLM输出的视觉token数量
        # - 计算资源：更多token需要更多计算，需要平衡性能和效率
        self.future_tokens = nn.Embedding(num_target_vision_tokens, hidden_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        
        # 位置编码
        if add_pos_embed:
            self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        # DiT块（支持交叉注意力）
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim,
                num_heads,
                mlp_ratio,
                dropout,
                use_cross_attention=use_cross_attention
            )
            for _ in range(num_layers)
        ])
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # VLM特征投影层（如果cross_attention_dim与hidden_dim不同）
        if cross_attention_dim is not None and cross_attention_dim != hidden_dim:
            self.vlm_projection = nn.Linear(cross_attention_dim, hidden_dim)
        else:
            self.vlm_projection = None
        
        # 动作解码器（预测速度场）
        self.action_decoder = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim
        )
        
        # Beta分布（用于时间步采样）
        self.beta_dist = Beta(noise_beta_alpha, noise_beta_beta)
        self.noise_s = noise_s
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def _normalize_states(self, states: torch.Tensor) -> torch.Tensor:
        """
        统一处理states维度，确保为[B, state_dim]
        
        Args:
            states: 机器人状态，可以是：
                - [B, state_dim]: 标准格式
                - [B, 1, state_dim]: 带时间步维度
                - [state_dim]: 单样本，会自动扩展为[1, state_dim]
                - [B, T, state_dim]: 多时间步，会取第一个时间步
        
        Returns:
            torch.Tensor: [B, state_dim]
        """
        if states.dim() == 1:
            # [state_dim] -> [1, state_dim]
            states = states.unsqueeze(0)
        elif states.dim() == 3:
            # [B, 1, state_dim] 或 [B, T, state_dim]
            if states.shape[1] == 1:
                states = states.squeeze(1)  # [B, 1, state_dim] -> [B, state_dim]
            else:
                # [B, T, state_dim] -> 取第一个时间步 [B, state_dim]
                states = states[:, 0, :]
        elif states.dim() == 4:
            # [B, 1, 1, state_dim] -> 压缩多余维度
            states = states.squeeze(2) if states.shape[2] == 1 else states.squeeze(1)
            if states.dim() == 3:
                states = states.squeeze(1) if states.shape[1] == 1 else states[:, 0, :]
        elif states.dim() != 2:
            raise ValueError(f"Unexpected states shape: {states.shape}, expected [B, state_dim] or [B, 1, state_dim]")
        
        return states
    
    def sample_time(self, batch_size, device, dtype):
        """采样时间步"""
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.noise_s - sample) / self.noise_s
    
    def forward(
        self,
        vlm_features: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        前向传播（训练模式）
        
        Args:
            vlm_features: VLM输出的特征 [B, seq_len, hidden_dim] 或 [B, hidden_dim]
            actions: 目标动作序列 [B, action_horizon, action_dim]（训练时需要）
            states: 机器人状态 [B, state_dim]（可选）
            encoder_attention_mask: 编码器注意力掩码 [B, seq_len]（可选）
            return_features: 是否返回中间特征（向后兼容）
            
        Returns:
            如果训练模式（actions提供）：
                loss: 标量损失值
            如果推理模式（actions为None）：
                actions: 预测的动作 [B, action_horizon, action_dim]
        """
        device = vlm_features.device
        batch_size = vlm_features.shape[0]
        
        # 处理vlm_features维度（用于交叉注意力）
        if vlm_features.dim() == 2:
            # [B, hidden_dim] -> [B, 1, hidden_dim]
            vlm_features_seq = vlm_features.unsqueeze(1)
        else:
            vlm_features_seq = vlm_features
        
        # 如果VLM特征维度与hidden_dim不同，进行投影
        # 注意：确保数据类型匹配（VLM可能输出bfloat16，但投影层使用float32）
        if self.vlm_projection is not None:
            # 转换数据类型以确保匹配（投影层通常使用float32）
            vlm_features_seq = vlm_features_seq.to(dtype=torch.float32)
            vlm_features_seq = self.vlm_projection(vlm_features_seq)
        
        # 训练模式：计算Flow Matching损失
        if actions is not None and self.training:
            # 采样噪声和时间步
            noise = torch.randn_like(actions)  # [B, action_horizon, action_dim]
            t = self.sample_time(batch_size, device, actions.dtype)  # [B]
            t_expanded = t[:, None, None]  # [B, 1, 1] 用于广播
            
            # 创建噪声轨迹
            noisy_trajectory = (1 - t_expanded) * noise + t_expanded * actions  # [B, action_horizon, action_dim]
            velocity = actions - noise  # [B, action_horizon, action_dim]
            
            # 离散化时间步
            t_discretized = (t * self.num_timestep_buckets).long()
            
            # 编码动作
            action_features = self.action_encoder(noisy_trajectory, t_discretized)  # [B, action_horizon, hidden_dim]
            
            # 编码状态（如果使用）
            if self.state_encoder is not None and states is not None:
                # 统一处理states维度：确保为[B, state_dim]
                states = self._normalize_states(states)
                
                state_features = self.state_encoder(states)  # [B, hidden_dim]
                state_features = state_features.unsqueeze(1)  # [B, 1, hidden_dim]
            else:
                state_features = None
            
            # 添加位置编码
            if self.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)  # [1, action_horizon, hidden_dim]
                action_features = action_features + pos_embs
            
            # 拼接状态、future tokens和动作特征
            # future_tokens作用详解：
            # 1. 可学习的占位符token，通过交叉注意力从VLM的hidden_states中提取视觉-语言信息
            # 2. 在训练过程中，这些token学习如何将VLM特征转换为动作预测的上下文
            # 3. 通过自注意力机制，future_tokens的信息会传播到action_features，影响动作编码
            # 4. 类似于Transformer中的特殊token（如[CLS]），但专门用于动作预测任务
            # 
            # 序列结构：[state_token(可选)] + [future_tokens] + [action_features]
            # 例如：num_target_vision_tokens=32, action_horizon=11, 有state时
            # 序列长度 = 1(state) + 32(future_tokens) + 11(actions) = 44
            # 
            # future_tokens如何求得：
            # - 初始化：使用nn.Embedding创建可学习的嵌入层，随机初始化（mean=0, std=0.02）
            # - 训练：通过反向传播自动学习，优化目标是使动作预测损失最小
            # - 推理：使用训练好的嵌入权重，从VLM特征中提取相关信息
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_target_vision_tokens, hidden_dim]
            
            # 拼接状态、future tokens和动作特征
            # 注意：state_features 已经在458行通过 unsqueeze(1) 处理为 [B, 1, hidden_dim] 格式
            # 这里不需要再次规范化，只需要防御性断言确保维度正确
            if state_features is not None:
                assert state_features.dim() == 3, f"state_features should be 3D [B, 1, hidden_dim], got {state_features.shape}"
                assert state_features.shape[1] == 1, f"state_features should have seq_len=1, got {state_features.shape[1]}"
                sa_embs = torch.cat([state_features, future_tokens, action_features], dim=1)  # [B, 1+num_tokens+action_horizon, hidden_dim]
            else:
                sa_embs = torch.cat([future_tokens, action_features], dim=1)  # [B, num_tokens+action_horizon, hidden_dim]
            
            # 通过DiT块
            # future_tokens作为query，通过交叉注意力关注VLM的hidden_states（key和value）
            # 这样future_tokens可以从VLM特征中提取相关信息，然后通过自注意力影响action_features
            x = sa_embs
            for block in self.blocks:
                x = block(
                    x,
                    encoder_hidden_states=vlm_features_seq,  # VLM特征作为交叉注意力的key和value
                    encoder_attention_mask=encoder_attention_mask
                )
            
            # 最终归一化
            x = self.final_norm(x)
            
            # 预测速度场（只取动作部分）
            pred = self.action_decoder(x)  # [B, seq_len, action_dim]
            pred_velocity = pred[:, -actions.shape[1]:]  # [B, action_horizon, action_dim]
            
            # 计算损失
            loss = ((pred_velocity - velocity) ** 2).mean()
            
            if return_features:
                return loss, x
            return loss
        
        # 推理模式：预测动作
        else:
            return self.predict_action(vlm_features, states, encoder_attention_mask)
    
    @torch.no_grad()
    def predict_action(
        self,
        vlm_features: torch.Tensor,
        states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        预测动作（推理模式）
        
        Args:
            vlm_features: VLM输出的特征 [B, seq_len, hidden_dim] 或 [B, hidden_dim]
            states: 机器人状态 [B, state_dim]（可选）
            encoder_attention_mask: 编码器注意力掩码 [B, seq_len]（可选）
            
        Returns:
            预测的动作 [B, action_horizon, action_dim]
        """
        device = vlm_features.device
        batch_size = vlm_features.shape[0]
        dtype = vlm_features.dtype
        
        # 处理vlm_features维度
        if vlm_features.dim() == 2:
            vlm_features_seq = vlm_features.unsqueeze(1)  # [B, 1, hidden_dim]
        else:
            vlm_features_seq = vlm_features
        
        # 如果VLM特征维度与hidden_dim不同，进行投影
        # 注意：确保数据类型匹配（VLM可能输出bfloat16，但投影层使用float32）
        if self.vlm_projection is not None:
            # 转换数据类型以确保匹配（投影层通常使用float32）
            vlm_features_seq = vlm_features_seq.to(dtype=torch.float32)
            vlm_features_seq = self.vlm_projection(vlm_features_seq)
        
        # 初始化动作（从噪声开始）
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=dtype,
            device=device
        )
        
        # 编码状态（如果使用）
        if self.state_encoder is not None and states is not None:
            # 统一处理states维度：确保为[B, state_dim]
            states = self._normalize_states(states)
            
            state_features = self.state_encoder(states)  # [B, hidden_dim]
            state_features = state_features.unsqueeze(1)  # [B, 1, hidden_dim]
        else:
            state_features = None
        
        # Euler积分去噪
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        
        for t_step in range(num_steps):
            t_cont = t_step / float(num_steps)  # 连续时间 [0, 1]
            t_discretized = int(t_cont * self.num_timestep_buckets)
            
            # 创建时间步张量
            timesteps_tensor = torch.full(
                size=(batch_size,),
                fill_value=t_discretized,
                device=device,
                dtype=torch.long
            )
            
            # 编码动作
            action_features = self.action_encoder(actions, timesteps_tensor)  # [B, action_horizon, hidden_dim]
            
            # 添加位置编码
            if self.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)  # [1, action_horizon, hidden_dim]
                action_features = action_features + pos_embs
            
            # 拼接状态、future tokens和动作特征
            # future_tokens作用详解：
            # 1. 可学习的占位符token，通过交叉注意力从VLM的hidden_states中提取视觉-语言信息
            # 2. 在训练过程中，这些token学习如何将VLM特征转换为动作预测的上下文
            # 3. 通过自注意力机制，future_tokens的信息会传播到action_features，影响动作编码
            # 4. 类似于Transformer中的特殊token（如[CLS]），但专门用于动作预测任务
            # 
            # 序列结构：[state_token(可选)] + [future_tokens] + [action_features]
            # 例如：num_target_vision_tokens=32, action_horizon=11, 有state时
            # 序列长度 = 1(state) + 32(future_tokens) + 11(actions) = 44
            # 
            # future_tokens如何求得：
            # - 初始化：使用nn.Embedding创建可学习的嵌入层，随机初始化（mean=0, std=0.02）
            # - 训练：通过反向传播自动学习，优化目标是使动作预测损失最小
            # - 推理：使用训练好的嵌入权重，从VLM特征中提取相关信息
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_target_vision_tokens, hidden_dim]
            
            # 拼接状态、future tokens和动作特征
            # 注意：state_features 已经在572行通过 unsqueeze(1) 处理为 [B, 1, hidden_dim] 格式
            # 这里不需要再次规范化，只需要防御性断言确保维度正确
            if state_features is not None:
                assert state_features.dim() == 3, f"state_features should be 3D [B, 1, hidden_dim], got {state_features.shape}"
                assert state_features.shape[1] == 1, f"state_features should have seq_len=1, got {state_features.shape[1]}"
                sa_embs = torch.cat([state_features, future_tokens, action_features], dim=1)  # [B, 1+num_tokens+action_horizon, hidden_dim]
            else:
                sa_embs = torch.cat([future_tokens, action_features], dim=1)  # [B, num_tokens+action_horizon, hidden_dim]
            
            # 通过DiT块
            # future_tokens作为query，通过交叉注意力关注VLM的hidden_states（key和value）
            # 这样future_tokens可以从VLM特征中提取相关信息，然后通过自注意力影响action_features
            x = sa_embs
            for block in self.blocks:
                x = block(
                    x,
                    encoder_hidden_states=vlm_features_seq,  # VLM特征作为交叉注意力的key和value
                    encoder_attention_mask=encoder_attention_mask
                )
            
            # 最终归一化
            x = self.final_norm(x)
            
            # 预测速度场
            pred = self.action_decoder(x)  # [B, seq_len, action_dim]
            pred_velocity = pred[:, -self.action_horizon:]  # [B, action_horizon, action_dim]
            
            # Euler积分更新动作
            actions = actions + dt * pred_velocity
        
        return actions

