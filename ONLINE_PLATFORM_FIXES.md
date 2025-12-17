# Online Platform Fixes Applied

## Issues Encountered on Online Platform

When submitting the project to the online grading platform, 4 tests failed due to differences between local and platform test expectations.

---

## ✅ **Fix 1: Dictionary Key Names in MultimodalDataset**

### **Problem:**
Tests expected dictionary keys `'text'` and `'image'`, but code was returning `'text_input'` and `'image_input'`.

### **Error Messages:**
```
test_multimodal_dataset_image_only:
AssertionError: Batch should contain image data
assert 'image' in {'image_input': array([...])}

test_multimodal_dataset_text_only:
AssertionError: Batch should contain text data
assert 'text' in {'text_input': array([...])}

test_multimodal_dataset_multimodal:
AssertionError: Batch should contain text data
assert 'text' in {'image_input': array([...])}
```

### **Solution:**
Updated `src/classifiers_mlp.py` line ~145-155 in `MultimodalDataset.__getitem__()`:

**Before:**
```python
if self.text_data is None:
    return {'image_input': image_batch}, label_batch
if self.image_data is None:
    return {'text_input': text_batch}, label_batch
else:
    return {'text_input': text_batch, 'image_input': image_batch}, label_batch
```

**After:**
```python
if self.text_data is None:
    return {'image': image_batch}, label_batch
if self.image_data is None:
    return {'text': text_batch}, label_batch
else:
    return {'text': text_batch, 'image': image_batch}, label_batch
```

### **Additional Changes:**
Updated Input layer names in `create_early_fusion_model()` function (line ~205):

**Before:**
```python
text_input = Input(shape=(text_input_size,), name='text_input')
image_input = Input(shape=(image_input_size,), name='image_input')
```

**After:**
```python
text_input = Input(shape=(text_input_size,), name='text')
image_input = Input(shape=(image_input_size,), name='image')
```

---

## ✅ **Fix 2: ConvNeXt Model Source (Hugging Face vs Keras)**

### **Problem:**
The online platform expected ConvNeXt models to be from **Hugging Face Transformers** (`TFConvNextV2Model`), but the code was using models from **TensorFlow Keras Applications** (`ConvNeXtTiny`).

### **Error Messages:**
```
test_foundational_cv_model_generic[convnextv2_tiny-TFConvNextV2Model-expected_output_shape1]:
AssertionError: Expected model class <class 'transformers.models.convnextv2.modeling_tf_convnextv2.TFConvNextV2Model'>, 
got <class 'keras.engine.functional.Functional'>
```

### **Root Cause:**
The project README explicitly states: **"You should at least implement [tensorflow ConvNextV2](https://huggingface.co/docs/transformers/en/model_doc/convnextv2#transformers.TFConvNextV2Model) from Hugging Face"**

The code was using `ConvNeXtTiny`, `ConvNeXtBase`, `ConvNeXtLarge` from `tensorflow.keras.applications`, but the platform tests check for `TFConvNextV2Model` from `transformers`.

### **Solution:**

#### **Step 1: Update Imports**
Changed `src/vision_embeddings_tf.py` line ~4:

**Before:**
```python
from transformers import TFViTModel, TFSwinModel
from tensorflow.keras.applications import (
    ResNet50, ResNet101, DenseNet121, DenseNet169, InceptionV3, 
    ConvNeXtTiny, ConvNeXtBase, ConvNeXtLarge
)
```

**After:**
```python
from transformers import TFViTModel, TFSwinModel, TFConvNextV2Model
from tensorflow.keras.applications import (
    ResNet50, ResNet101, DenseNet121, DenseNet169, InceptionV3
)
```

#### **Step 2: Load ConvNeXt from Hugging Face**
Updated `src/vision_embeddings_tf.py` lines ~111-119:

**Before:**
```python
elif backbone == 'convnextv2_tiny':
    self.base_model = ConvNeXtTiny(weights='imagenet', include_top=False, input_tensor=input_layer)
```

**After:**
```python
elif backbone == 'convnextv2_tiny':
    # Load ConvNeXt V2 from Hugging Face transformers (as per project requirements)
    self.base_model = TFConvNextV2Model.from_pretrained("facebook/convnextv2-tiny-1k-224", use_safetensors=False)
elif backbone == 'convnextv2_base':
    self.base_model = TFConvNextV2Model.from_pretrained("facebook/convnextv2-base-1k-224", use_safetensors=False)
elif backbone == 'convnextv2_large':
    self.base_model = TFConvNextV2Model.from_pretrained("facebook/convnextv2-large-1k-224", use_safetensors=False)
```

**Note:** `use_safetensors=False` is required due to a compatibility issue between transformers library v4.57.0 and safetensors format.

#### **Step 3: Handle Channels-First Format**
ConvNeXtV2 from Hugging Face requires **channels-first** format (batch, channels, height, width) instead of channels-last.

Updated `src/vision_embeddings_tf.py` lines ~147-162:

**Before:**
```python
if backbone in ['vit_base', 'vit_large', 'convnextv2_tiny', 'convnextv2_base', 'convnextv2_large', 'swin_tiny', 'swin_small', 'swin_base']:
    outputs = self.base_model(input_layer).pooler_output
    self.model = Model(inputs=input_layer, outputs=outputs)
```

**After:**
```python
# ConvNeXtV2 from Hugging Face requires channels-first and Lambda wrapper
if backbone in ['convnextv2_tiny', 'convnextv2_base', 'convnextv2_large']:
    # Transpose from channels-last (H,W,C) to channels-first (C,H,W)
    input_layer_transposed = Lambda(lambda x: tf.transpose(x, perm=[0, 3, 1, 2]))(input_layer)
    # Wrap HF model in Lambda layer (required for Keras Functional API)
    outputs = Lambda(lambda x: self.base_model(x).pooler_output, output_shape=(768,))(input_layer_transposed)
    self.model = Model(inputs=input_layer, outputs=outputs)
# ViT and Swin from Hugging Face (channels-last compatible)
elif backbone in ['vit_base', 'vit_large', 'swin_tiny', 'swin_small', 'swin_base']:
    outputs = Lambda(lambda x: self.base_model(x).pooler_output, output_shape=(768,))(input_layer)
    self.model = Model(inputs=input_layer, outputs=outputs)
```

#### **Step 4: Fix All Transformers Loading**
Added `use_safetensors=False` to all Hugging Face model loading calls:

```python
# ViT models
self.base_model = TFViTModel.from_pretrained(backbone_path[backbone], use_safetensors=False)

# Swin models  
self.base_model = TFSwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224", use_safetensors=False)
```

### **Key Insights:**

1. **Hugging Face models cannot be called directly** in Keras Functional API - they require Lambda layer wrapping
2. **ConvNeXtV2 expects channels-first** format (batch, C, H, W) - must transpose from channels-last (batch, H, W, C)
3. **Lambda layers need output_shape** specified when shape inference fails
4. **safetensors compatibility issue** - use `use_safetensors=False` parameter with transformers v4.57.0

---

## ✅ **Fix 3: Updated Local Tests**

Updated `tests/test_classifiers_mlp.py` to match the platform expectations:

**Changed in 3 test functions:**
- `test_multimodal_dataset_image_only` (line ~84)
- `test_multimodal_dataset_text_only` (line ~111)
- `test_multimodal_dataset_multimodal` (line ~138)

**Before:**
```python
assert 'image_input' in batch_inputs
assert 'text_input' in batch_inputs
```

**After:**
```python
assert 'image' in batch_inputs
assert 'text' in batch_inputs
```

---

## ✅ **Test Results After Fixes**

**Local Tests:** ✅ 18/18 passing (100%)

**Expected Online Platform Results:**
- ✅ test_multimodal_dataset_image_only - FIXED
- ✅ test_multimodal_dataset_text_only - FIXED
- ✅ test_multimodal_dataset_multimodal - FIXED
- ✅ test_foundational_cv_model_generic[convnextv2_tiny...] - FIXED

---

## 📝 **Summary**

All 4 failing tests were caused by naming convention differences between local and platform expectations. The fixes ensure compatibility with both environments by:

1. Using standardized dictionary keys (`'text'` and `'image'`) for data batches
2. Correctly categorizing TensorFlow Keras models vs Hugging Face transformers
3. Updating local tests to match platform expectations

**Project Status:** Ready for resubmission to online platform ✅
