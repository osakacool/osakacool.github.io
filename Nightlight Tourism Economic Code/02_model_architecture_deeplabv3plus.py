import tensorflow as tf
from tensorflow.keras import layers, models

def ASPP(inputs, rates=[6, 12, 18]):
    """Atrous Spatial Pyramid Pooling (Table 2)"""
    # 1x1 Conv
    x1 = layers.Conv2D(256, 1, padding='same', use_bias=False)(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    
    # Atrous Convolutions
    aspp_layers = [x1]
    for rate in rates:
        x = layers.Conv2D(256, 3, padding='same', dilation_rate=rate, use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        aspp_layers.append(x)
        
    # Image Pooling
    img_pool = layers.GlobalAveragePooling2D(keepdims=True)(inputs)
    img_pool = layers.Conv2D(256, 1, padding='same', use_bias=False)(img_pool)
    img_pool = layers.BatchNormalization()(img_pool)
    img_pool = layers.ReLU()(img_pool)
    img_pool = layers.UpSampling2D(size=(inputs.shape[1]//img_pool.shape[1], inputs.shape[2]//img_pool.shape[2]), interpolation='bilinear')(img_pool)
    aspp_layers.append(img_pool)
    
    return layers.Concatenate()(aspp_layers)

def build_deeplabv3plus_5ch(input_shape=(512, 512, 5), num_classes=5):
    """
    DeepLabV3+ with ResNet-50 backbone modified for 5-channel input.
    Corresponds to Table 2 Configuration.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Backbone: ResNet50 (Pre-trained on ImageNet, but we adapt the first conv for 5 channels)
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    # 注意：由于输入是5通道，需要自定义第一层卷积权重初始化，此处用代码逻辑示意
    # 实际中通常复制RGB权重并初始化另外2通道
    
    # 提取低层特征 (Low-level features) 用于 Decoder
    low_level_feat = base_model.get_layer('conv2_block3_out').output 
    
    # 提取高层特征 (High-level features)
    high_level_feat = base_model.get_layer('conv4_block6_out').output
    
    # ASPP Module
    aspp_out = ASPP(high_level_feat)
    x = layers.Conv2D(256, 1, padding='same', use_bias=False)(aspp_out)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Decoder Module (1x1 Conv + Bilinear Upsampling)
    low_level = layers.Conv2D(48, 1, padding='same', use_bias=False)(low_level_feat)
    low_level = layers.BatchNormalization()(low_level)
    low_level = layers.ReLU()(low_level)
    
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    x = layers.Concatenate()([x, low_level])
    
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Output Layer (Softmax Activation for 5 classes)
    outputs = layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def weighted_categorical_crossentropy(y_true, y_pred, class_weights):
    """Table 2: Weighted Categorical Cross-Entropy Loss"""
    # class_weights: inverse frequency weights
    weights = tf.constant(class_weights, dtype=tf.float32)
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    # 应用类别权重
    weighted_loss = loss * tf.reduce_sum(y_true * weights, axis=-1)
    return tf.reduce_mean(weighted_loss)

if __name__ == "__main__":
    model = build_deeplabv3plus_5ch()
    model.summary()