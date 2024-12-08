文書を日本語に翻訳させていただきます。

## 目次
1. [Batchnorm Forward](#Batchnorm Forward)
2. [Batchnorm Backward](#Batchnorm Backward)
3. [Batchnorm Finalize](#Batchnorm Finalize)
4. [BGenerate Stats](#Generate Stats)

### Batchnorm Forward
Batchnorm演算は以下を計算します：
```math
output = scale*{input - mean \over \sqrt{variance + epsilon}} + bias
```

オプションとして、以下も計算します：
```math
next\_running\_mean = (1 - momentum)*previous\_running\_mean + momentum*current\_running\_mean
```
```math
next\_running\_variance = (1 - momentum)*previous\_running\_variance + momentum*current\_running\_variance
```

上記の方程式を実現するAPIは以下の通りです：
```
std::array<std::shared_ptr<Tensor_attributes>, 5> batchnorm(std::shared_ptr<Tensor_attributes>& input,
                                                            std::shared_ptr<Tensor_attributes>& scale,
                                                            std::shared_ptr<Tensor_attributes>& bias,
                                                            Batchnorm_attributes attributes); 
```
出力配列には、以下の順序でテンソルが含まれます：`[output, saved_mean, saved_invariance, next_running_mean, next_running_variance]`

Batchnorm_attributesは、オプションの入力テンソルやその他の演算属性を提供するための軽量な構造体で、以下のsetterを持ちます：
```
Batchnorm_attributes&
set_previous_running_stats(std::shared_ptr<Tensor_attributes>& previous_running_mean,
                            std::shared_ptr<Tensor_attributes>& previous_running_variance,
                            std::shared_ptr<Tensor_attributes>& momentum)

Batchnorm_attributes&
set_name(std::string const&)

Batchnorm_attributes&
set_compute_data_type(DataType_t value)
```

Python API: 
- batchnorm
    - input
    - scale
    - bias
    - in_running_mean
    - in_running_var
    - epsilon
    - momentum
    - compute_data_type
    - name

### Batchnorm Finalize

`bn_finalize`は、genstat演算によって生成された統計から、次のイテレーションに必要な統計を計算します。
```
    std::array<std::shared_ptr<Tensor_attributes>, 6> bn_finalize(std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  std::shared_ptr<Tensor_attributes>,
                                                                  BN_finalize_attributes);
```

出力は以下の順序です：`[EQ_SCALE, EQ_BIAS, MEAN, INV_VARIANCE, NEXT_RUNNING_MEAN, NEXT_RUNNING_VAR]`

### Batchnorm Backward(DBN)
DBN演算は、batchnorm forward演算のバックプロパゲーション時にデータ勾配、スケール勾配、バイアス勾配を計算します。

これを実現するAPIは以下の通りです：
```
std::array<std::shared_ptr<Tensor_attributes>, 3> batchnorm_backward(std::shared_ptr<Tensor_attributes> loss,
                                                                         std::shared_ptr<Tensor_attributes> input,
                                                                         std::shared_ptr<Tensor_attributes> scale,
                                                                         Batchnorm_backward_attributes);
```
出力配列には、以下の順序でテンソルが含まれます：`[入力勾配, スケール勾配, バイアス勾配]`

DBN属性は以下のsetterを持つ軽量な構造体です：
```
Batchnorm_backward_attributes&
set_saved_mean_and_inv_variance(std::shared_ptr<Tensor_attributes> saved_mean,
                                std::shared_ptr<Tensor_attributes> saved_inverse_variance)
                                
Batchnorm_backward_attributes&
set_epsilon(std::shared_ptr<Tensor_attributes> epsilon)

Batchnorm_backward_attributes&
set_name(std::string const&)

Batchnorm_backward_attributes&
set_compute_data_type(DataType_t value)
```
(saved meanとinverse_variance)または(epsilon)のいずれかの設定のみが必要です。

### Generate Stats
Genstats演算は、チャンネル次元ごとの合計と二乗和を計算します。

これを実現するAPIは以下の通りです：
```
std::array<std::shared_ptr<Tensor_attributes>, 2>
cudnn_frontend::graph::genstats(std::shared_ptr<Tensor_attributes>, Genstats_attributes);
```
出力配列には、以下の順序でテンソルが含まれます：`[sum, square_sum]`

Genstats属性は以下のsetterを持つ軽量な構造体です：
```
Genstats_attributes&
set_name(std::string const&)

Genstats_attributes&
set_compute_data_type(DataType_t value)
```

Python API: 
- genstats
    - input
    - compute_data_type
    - name

### Layernorm Forward

Layer normは以下を計算します：

```match
output = scale*{input - mean \over \sqrt{variance + epsilon}} + bias
```

ここで、正規化は各サンプルに対して独立して特徴量全体で行われます。

上記の方程式を実現するAPIは以下の通りです：
```
std::array<std::shared_ptr<Tensor_attributes>, 3> layernorm(std::shared_ptr<Tensor_attributes>& input,
                                                            std::shared_ptr<Tensor_attributes>& scale,
                                                            std::shared_ptr<Tensor_attributes>& bias,
                                                            Layernorm_attributes attributes); 
```
出力配列には、以下の順序でテンソルが含まれます：`[output, mean, variance]`

Layernorm_attributesは、オプションの入力テンソルやその他の演算属性を提供するための軽量な構造体で、以下のsetterを持ちます：
```
Layernorm_attributes&
set_name(std::string const&)

Layernorm_attributes&
set_compute_data_type(DataType_t value)
```

Python API: 
- layernorm
    - norm_forward_phase
    - input
    - scale
    - bias
    - epsilon
    - compute_data_type
    - name

### Layernorm Backward

DLN演算は、layernorm forward演算のバックプロパゲーション時にデータ勾配、スケール勾配、バイアス勾配を計算します。

これを実現するAPIは以下の通りです：
```
std::array<std::shared_ptr<Tensor_attributes>, 3>
            layernorm_backward(std::shared_ptr<Tensor_attributes> dy,
                          std::shared_ptr<Tensor_attributes> x,
                          std::shared_ptr<Tensor_attributes> scale,
                          Layernorm_backward_attributes options);
```
出力配列には、以下の順序でテンソルが含まれます：`[入力勾配, スケール勾配, バイアス勾配]`

Layernorm_attributesは、オプションの入力テンソルやその他の演算属性を提供するための軽量な構造体で、以下のsetterを持ちます：
```
Layernorm_attributes&
set_name(std::string const&)

Layernorm_attributes&
set_compute_data_type(DataType_t value)
```

Python API: 
- layernorm
    - input
    - scale
    - loss
    - compute_data_type
    - name

### Instancenorm Forward

Instance normは以下を計算します：

$$ output = scale*{input - mean \over \sqrt{variance + epsilon}} + bias $$

ここで、正規化は各サンプルに対して行われます。

上記の方程式を実現するAPIは以下の通りです：
```
std::array<std::shared_ptr<Tensor_attributes>, 3> instancenorm(std::shared_ptr<Tensor_attributes>& input,
                                                            std::shared_ptr<Tensor_attributes>& scale,
                                                            std::shared_ptr<Tensor_attributes>& bias,
                                                            Instancenorm_attributes attributes); 
```
出力配列には、以下の順序でテンソルが含まれます：`[output, mean, variance]`

Instancenorm_attributesは、オプションの入力テンソルやその他の演算属性を提供するための軽量な構造体で、以下のsetterを持ちます：
```
Instancenorm_attributes&
set_name(std::string const&)

Instancenorm_attributes&
set_compute_data_type(DataType_t value)
```

Python API: 
- instancenorm
    - norm_forward_phase
    - input
    - scale
    - bias
    - epsilon
    - compute_data_type
    - name

### Instancenorm Backward

DIN演算は、instancenorm forward演算のバックプロパゲーション時にデータ勾配、スケール勾配、バイアス勾配を計算します。

これを実現するAPIは以下の通りです：
```
std::array<std::shared_ptr<Tensor_attributes>, 3>
            instancenorm_backward(std::shared_ptr<Tensor_attributes> dy,
                          std::shared_ptr<Tensor_attributes> x,
                          std::shared_ptr<Tensor_attributes> scale,
                          Instancenorm_backward_attributes options);
```
出力配列には、以下の順序でテンソルが含まれます：`[入力勾配, スケール勾配, バイアス勾配]`

Instancenorm_attributesは、オプションの入力テンソルやその他の演算属性を提供するための軽量な構造体で、以下のsetterを持ちます：
```
Instancenorm_attributes&
set_name(std::string const&)

Instancenorm_attributes&
set_compute_data_type(DataType_t value)
```

Python API: 
- layernorm
    - input
    - scale
    - loss
    - compute_data_type
    - name
