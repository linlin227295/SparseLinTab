"""Contains Tabular Transformer Model definition."""
from itertools import cycle

import torch
import torch.nn as nn

from sparselintab.model.image_patcher import LinearImagePatcher
from sparselintab.model.sparselintab_modules import MHSA
from sparselintab.model.sparselintab_modules import get_EF
from sparselintab.utils.config_utils import Args
from sparselintab.utils.encode_utils import torch_cast_to_dtype

IMAGE_PATCHER_SETTING_TO_CLASS = {
    'linear': LinearImagePatcher,
}


class sparselintabModel(nn.Module):
    """Non-Parametric Transformers.

    Applies Multi-Head Self-Attention blocks between datapoints,
    and within each datapoint.

    For all model variants, we expect a list of input data, `X_ragged`:
    ```
        len(X_ragged) == N
        X_ragged[i].shape == (D, H_i)
    ```
    In other words, we have `N` input samples. All samples share the same
    number of `D` features, where each feature i is encoded in `H_i`
    dimensions. "Encoding" here refers to the data preprocessing, i.e. the
    one-hot-encoding for categorical features, as well as well as adding
    the mask tokens. (Note that this is done by the code and the user is
    expected to provide datasets as given in `sparselintab.data_loaders`.)

    High-level model overview:

    Initially in sparselintabModel, `self.in_embedding()` linearly embeds each of the
    `D` feature columns to a shared embedding dimension `E`.
    We learn separate embedding weights for each column.
    This allows us to obtain the embedded data matrix `X_emb` as a
    three-dimensional tensor of shape `(N, D, E)`.
    `E` is referred to as `dim_feat` in the code below.

    After embedding the data, we apply sparselintab.
    See `build_sparselintab()` for further information.
    sparselintab applies a series of attention blocks on the input.

    We eventually obtain output of shape `(N, D, E)`,
    which is projected back to the dimensions of the input `X_ragged` using
    `self.out_embedding()`, which applies a learned linear embedding to
    each column `D` separately.
    """

    def __init__(self, c, metadata, device=None):
        """Initialise sparselintabModel.

        Args:
            c: wandb config
            metadata: Dict, from which we retrieve:
                input_feature_dims: List[int], used to specify the number of
                    one-hot encoded dimensions for each feature in the table
                    (used when reloading models from checkpoints).
                cat_features: List[int], indices of categorical features, used
                    in model initialization if using feature type embeddings.
                num_features: List[int], indices of numerical features, used
                    in model initialization if using feature type embeddings.
                cat_target_cols: List[int], indices of categorical target
                    columns, used if there is a special embedding dimension
                    for target cols.
                num_target_cols: List[int], indices of numerical target
                    columns, used if there is a special embedding dimension
                    for target cols.
            device: Optional[int].
        """
        super().__init__()

        # *** Extract Configs ***提取配置
        # cannot deepcopy wandb config.
        if c.mp_distributed:
            self.c = Args(c.__dict__)
        else:
            self.c = Args(c)

        # * Main model configuration *
        self.device = device

        # * Dataset Metadata *数据集元数据
        input_feature_dims = metadata['input_feature_dims']
        cat_features = metadata['cat_features']
        num_features = metadata['num_features']
        cat_target_cols = metadata['cat_target_cols']
        num_target_cols = metadata['num_target_cols']

        # * Dimensionality Configs *
        # how many attention blocks are stacked after each other
        self.stacking_depth = c.model_stacking_depth

        # the shared embedding dimension of each attribute is given by
        self.dim_hidden = c.model_dim_hidden

        # we use num_heads attention heads
        self.num_heads = c.model_num_heads

        # Linformer Configs
        self.dimk_row = c.dimk_row
        self.dimk_col = c.dimk_col
        self.method = c.method
        self.parameter_sharing = c.parameter_sharing
        if c.exp_batch_size == -1:
            self.N = metadata['N']
        else:
            self.N = c.exp_batch_size

        # how many feature columns are in the input data
        # apply image patching if specified 图像处理与特征列处理
        if self.c.model_image_n_patches:
            extra_args = {}

            # num_input_features = n_patches per image
            self.image_patcher = IMAGE_PATCHER_SETTING_TO_CLASS[
                self.c.model_image_patch_type](
                input_feature_dims=input_feature_dims,
                dim_hidden=self.dim_hidden,
                c=c, **extra_args)
            sparselintab_attrs = self.image_patcher.get_sparselintab_attrs()
            for k, v in sparselintab_attrs.items():
                self.__setattr__(name=k, value=v)
        else:
            self.image_patcher = None
            self.num_input_features = len(input_feature_dims)

        # whether or not to add a feature type embedding 特征类型嵌入
        self.use_feature_type_embedding = c.model_feature_type_embedding

        # whether or not to add a feature index embedding 特征索引嵌入
        self.use_feature_index_embedding = c.model_feature_index_embedding

        # *** Build Model ***

        # We immediately embed each element
        # (i.e., a table with N rows and D columns has N x D elements)
        # to the hidden_dim. Similarly, in the output, we will "de-embed"
        # from this hidden_dim.

        # Build encoder  构建编码器
        self.enc = self.get_sparselintab()

        # *** Dropout and LayerNorm in In-/Out-Embedding ***

        # Hidden dropout is applied for in- and out-embedding  dropout层
        self.embedding_dropout = (
            nn.Dropout(p=c.model_hidden_dropout_prob)
            if c.model_hidden_dropout_prob else None)

        # LayerNorm applied after embedding, before dropout  层归一化
        if self.c.model_embedding_layer_norm and device is None:
            print(
                'Must provide a device in sparselintab initialization with embedding '
                'LayerNorm.')
        elif self.c.model_embedding_layer_norm:
            # we batch over rows and columns
            # (i.e. just normalize over E)
            layer_norm_dims = [self.dim_hidden]
            self.embedding_layer_norm = nn.LayerNorm(
                layer_norm_dims, eps=self.c.model_layer_norm_eps)
        else:
            self.embedding_layer_norm = None

        # *** Input In/Out Embeddings ***
        # Don't use for Image Patching - those are handled by the respective
        # init_image_patching

        # In-Embedding
        # Linearly embeds each of the `D` [len(input_feature_dims)] feature
        # columns to a shared embedding dimension E [dim_hidden].
        # Before the embedding, each column has its own dimensionionality
        # H_j [dim_feature_encoding], given by the encoding dimension of the
        # feature (e.g. This is given by the one-hot-encoding size for
        # categorical variables + one dimension for the mask token and two-
        # dimensional for continuous variables (scalar + mask_token)).
        # See docstring of sparselintabModel for further context.

        if self.image_patcher is None:
            self.in_embedding = nn.ModuleList([  # 对于 input_feature_dims 中的每个 dim_feature_encoding 值，都会创建一个新的 Linear 层，进行Embedding
                nn.Linear(dim_feature_encoding, self.dim_hidden)
                for dim_feature_encoding in input_feature_dims])

        # Feature Type Embedding 特征类型嵌入
        # 可选地，我们构建“特征类型”嵌入 - 即我们根据特征是 (i) 数字还是 (ii) 分类来学习表示。
        # Optionally, we construct "feature type" embeddings -- i.e. we learn a
        # representation based on whether the feature is either
        # (i) numerical or (ii) categorical.
        if self.use_feature_type_embedding:
            if cat_features is None or num_features is None:
                raise Exception(
                    'Must provide cat_feature and num_feature indices at '
                    'sparselintab initialization if you aim to compute feature type'
                    ' embeddings.')

            if c.mp_distributed and device is None:
                raise Exception(
                    'Must provide device to sparselintab initialization: in '
                    'distributed setting, and aim to do feature type '
                    'embedding.')

            # If all features are either categorical or numerical,
            # don't bother.
            if len(cat_features) == 0 or len(num_features) == 0:
                print(
                    'All features are either categorical or numerical. '
                    'Not going to bother doing feature type embeddings.')
                self.feature_type_embedding = None
            else:
                self.feature_types = torch_cast_to_dtype(torch.empty(  # 创建feature_types，指示哪些是数值特征哪些是分类特征
                    self.num_input_features, device=device), 'long')

                for feature_index in range(self.num_input_features):
                    if feature_index in num_features:
                        self.feature_types[feature_index] = 0
                    elif feature_index in cat_features:
                        self.feature_types[feature_index] = 1
                    else:
                        raise Exception

                self.feature_type_embedding = nn.Embedding(  # 将 self.feature_types 中的 0 和 1 映射到一个 dim_hidden 维度的向量空间
                    2, self.dim_hidden)

            print(
                f'Using feature type embedding (unique embedding for '
                f'categorical and numerical features).')
        else:
            self.feature_type_embedding = None

        # Feature Index Embedding
        # Optionally, learn a representation based on the index of the column.
        # Allows us to explicitly encode column identity, as opposed to
        # producing this indirectly through the per-column feature embeddings.
        if self.use_feature_index_embedding:
            if c.mp_distributed and device is None:
                raise Exception(
                    'Must provide device to sparselintab initialization: in '
                    'distributed setting, and aim to do feature index '
                    'embedding.')

            self.feature_indices = torch_cast_to_dtype(
                torch.arange(self.num_input_features, device=device), 'long')

            self.feature_index_embedding = nn.Embedding(  # 将每个特征的索引映射到一个 dim_hidden 维度的向量中，为神经网络提供特征级别的信息表示
                self.num_input_features, self.dim_hidden)

            print(
                f'Using feature index embedding (unique embedding for '
                f'each column).')
        else:
            self.feature_index_embedding = None

        # Out embedding. 输出嵌入
        # The outputs of the AttentionBlocks have shape (N, D, E)
        # [N, len(input_feature_dim), dim_hidden].
        # For each of the column j, we then project back to the dimensionality
        # of that column in the input (N, H_j-1), subtracting 1, because we do
        # not predict the mask tokens, which were present in the input.

        if self.image_patcher is None:
            # Need to remove the mask column if we are using BERT augmentation,
            # otherwise we just project to the same size as the input.
            if self.c.model_bert_augmentation:
                get_dim_feature_out = lambda x: x - 1
            else:
                get_dim_feature_out = lambda x: x

            self.out_embedding = nn.ModuleList([
                nn.Linear(
                    self.dim_hidden,
                    get_dim_feature_out(dim_feature_encoding))
                for dim_feature_encoding in input_feature_dims])

        # *** Gradient Clipping ***梯度裁剪防止梯度爆炸
        if c.exp_gradient_clipping:
            clip_value = c.exp_gradient_clipping
            print(f'Clipping gradients to value {clip_value}.')
            for p in self.parameters():  # 这是一个迭代器，返回模型的所有参数
                p.register_hook(  # register_hook 方法为每个参数 p 注册一个钩子函数。钩子函数会在反向传播计算梯度后调用
                    lambda grad: torch.clamp(grad, -clip_value, clip_value))  # 对每个参数的梯度 grad 进行裁剪

    def get_sparselintab(self):
        """
        A model performing "flattened" attention over the rows and
        "nested" attention over the columns.

        This is reasonable if we don't aim to maintain column equivariance
        (which we essentially never do, because of the column-specific
        feature embeddings at the input and output of the sparselintab encoder).

        This is done by concatenating the feature outputs of column
        attention and inputting them to row attention. Therefore, it requires
        reshaping between each block, splitting, and concatenation.
        “flattened” attention：指的是对行执行的注意力，即在行上“展开”数据并执行注意力操作。
        “nested” attention：指的是对列的注意力操作，即将列视作独立的特征维度，单独执行注意力操作。
        """
        if self.stacking_depth < 2:  # 堆叠深度检查
            raise ValueError(
                f'Stacking depth {self.stacking_depth} invalid.'
                f'Minimum stacking depth is 2.')
        if self.stacking_depth % 2 != 0:
            raise ValueError('Please provide an even stacking depth.')

        print('Building sparselintab.')

        # *** Construct arguments for row and column attention. ***构建行列注意力的参数

        row_att_args = {'c': self.c}
        col_att_args = {'c': self.c}

        # Perform attention over rows first
        att_args = cycle([row_att_args, col_att_args])  # 使用 itertools.cycle 创建一个无限循环的参数迭代器，使得后续的堆叠层可以交替从行注意力参数和列注意力参数中取值。
        AttentionBlocks = cycle([MHSA])  # 同样使用 cycle 循环生成多头自注意力（MHSA）模块

        D = self.num_input_features  # 输入数据的特征列数

        enc = []

        if self.c.model_hybrid_debug:  # 调试信息
            enc.append(Print())

        # Reshape to flattened representation (1, N, D*dim_input)   重塑为扁平表示（1，N，D*dim_input）
        enc.append(ReshapeToFlat())

        enc = self.build_hybrid_enc(  # 构建编码器
            enc, AttentionBlocks, att_args, D)

        enc = nn.Sequential(*enc)  # 将编码器构建为nn.sequential模块
        return enc

    def build_hybrid_enc(self, enc, AttentionBlocks, att_args, D):
        final_shape = None

        if self.c.model_hybrid_debug:  # 是否为调试模式
            stack = [Print()]
        else:
            stack = []

        layer_index = 0

        head_dim = self.dim_hidden // self.num_heads
        # 获取投影矩阵
        E_proj_row = get_EF(self.N, self.dimk_row, self.method, head_dim)
        E_proj_col = get_EF(D, self.dimk_col, self.method, head_dim)

        while layer_index < self.stacking_depth:  # 迭代堆叠层  交替执行行和列的注意力机制
            if layer_index % 2 == 1:  # 奇数层（列注意力）
                # Input is already in nested shape (N, D, E)
                if self.parameter_sharing != 'layerwise':  # 每一层都不一样
                    E_proj_col = get_EF(D, self.dimk_col, self.method, head_dim)
                stack.append(next(AttentionBlocks)(
                    self.dim_hidden, self.dim_hidden, self.dim_hidden, E_proj_col, self.dimk_col, D,
                    **next(att_args)))

                # Reshape to flattened representation
                stack.append(ReshapeToFlat())  # （1,N,D*E)
                final_shape = 'flat'

                if self.c.model_hybrid_debug:
                    stack.append(Print())
            else:
                # Input is already in flattened shape (1, N, D*E)
                # 偶数层（行注意力)
                # Attend between instances N
                # whenever we attend over the instances,
                # we consider dim_hidden = self.c.dim_hidden * D
                if self.parameter_sharing != 'layerwise':  # 每一层都不一样
                    E_proj_row = get_EF(self.N, self.dimk_row, self.method, head_dim)
                stack.append(next(AttentionBlocks)(
                    self.dim_hidden * D, self.dim_hidden * D,
                    self.dim_hidden * D, E_proj_row, self.dimk_row, self.N,
                    **next(att_args)))

                # Reshape to nested representation
                stack.append(ReshapeToNested(D=D))  # (N,D,E)
                final_shape = 'nested'

                if self.c.model_hybrid_debug:
                    stack.append(Print())

            # Conglomerate the stack into the encoder thus far
            enc += stack
            stack = []

            layer_index += 1

        # Reshape to nested representation, for correct treatment
        # after enc
        if final_shape == 'flat':
            enc.append(ReshapeToNested(D=D))  # (N,D,E)

        return enc

    def forward(self, X_ragged, X_labels=None, eval_model=None):
        if self.image_patcher is not None:
            X = self.image_patcher.encode(X_ragged)
            in_dims = [X.size(0), X.size(1), -1]
        else:  # 输入数据处理
            in_dims = [X_ragged[0].shape[0], len(X_ragged), -1]

            # encode ragged input array D x {(NxH_j)}_j to NxDxE)
            X = [embed(X_ragged[i]) for i, embed in enumerate(self.in_embedding)]
            X = torch.stack(X, 1)  # N*D*E

        # Compute feature type (cat vs numerical) embeddings, and add them  添加特征类型嵌入
        if self.feature_type_embedding is not None:
            feature_type_embeddings = self.feature_type_embedding(
                self.feature_types)  # D*E

            # Add a batch dimension (the rows)
            feature_type_embeddings = torch.unsqueeze(
                feature_type_embeddings, 0)  # 1*D*E

            # Tile over the rows
            feature_type_embeddings = feature_type_embeddings.repeat(
                X.size(0), 1, 1)  # N*D*E

            # Add to X
            X = X + feature_type_embeddings

        # Compute feature index embeddings, and add them    添加特征索引嵌入
        if self.feature_index_embedding is not None:
            feature_index_embeddings = self.feature_index_embedding(
                self.feature_indices)  # D*E

            # Add a batch dimension (the rows)
            feature_index_embeddings = torch.unsqueeze(
                feature_index_embeddings, 0)  # 1*D*E

            # Tile over the rows
            feature_index_embeddings = feature_index_embeddings.repeat(
                X.size(0), 1, 1)  # N*D*E

            # Add to X
            X = X + feature_index_embeddings

        # Embedding tensor currently has shape (N x D x E)

        # Follow BERT in applying LayerNorm -> Dropout on embeddings 嵌入层归一化和dropout
        if self.embedding_layer_norm is not None:
            X = self.embedding_layer_norm(X)

        if self.embedding_dropout is not None:
            X = self.embedding_dropout(X)

        # apply sparselintab
        X = self.enc(X)

        if X.shape[1] == in_dims[0]:  # 调整输出维度（如果需要）
            # for uneven stacking_depth, need to permute one last time
            # to obtain output of shape (N, D, E)
            X = X.permute([1, 0, 2])

        # Dropout before final projection (follows BERT, which performs
        # dropout before e.g. projecting to logits for sentence classification)
        if self.embedding_dropout is not None:  # 最后的 Dropout 处理
            X = self.embedding_dropout(X)

        if self.image_patcher is None:  # 还原到 Ragged 格式或图像格式
            # project back to ragged (dimensions D x {(NxH_j)}_j )
            # Is already split up across D
            X_ragged = [de_embed(X[:, i]) for i, de_embed in enumerate(
                self.out_embedding)]
        else:
            X_ragged = self.image_patcher.decode(X)

        return X_ragged


class Permute(nn.Module):
    """Permutation as nn.Module to include in nn.Sequential."""

    def __init__(self, idxs):
        super(Permute, self).__init__()
        self.idxs = idxs

    def forward(self, X):
        return X.permute(self.idxs)


class ReshapeToFlat(nn.Module):
    """Reshapes a tensor of shape (N, D, E) to (1, N, D*E)."""

    def __init__(self):
        super(ReshapeToFlat, self).__init__()

    @staticmethod
    def forward(X):
        return X.reshape(1, X.size(0), -1)


class ReshapeToNested(nn.Module):
    """Reshapes a tensor of shape (1, N, D*E) to (N, D, E)."""

    def __init__(self, D):
        super(ReshapeToNested, self).__init__()
        self.D = D

    def forward(self, X):
        return X.reshape(X.size(1), self.D, -1)


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print('Debug', x.shape)
        return x
