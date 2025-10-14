# fix_huggingface_models.py - Generalized version for any Hugging Face model
import os
import shutil
import subprocess
import sys
import argparse

def clean_system(model_name=None):
    """Clean system for Hugging Face model installation"""
    print(f"🧹 Cleaning system for {model_name or 'Hugging Face'} installation...")
    
    # Remove model-specific directories
    conflicts = [
        model_name, 
        f'./{model_name}', 
        f'../{model_name}',
        f'models--{model_name.replace("/", "--")}' if model_name else None
    ]
    
    for conflict in conflicts:
        if conflict and os.path.exists(conflict):
            shutil.rmtree(conflict)
            print(f"Removed: {conflict}")
    
    # Clear caches
    caches = [
        os.path.expanduser('~/.cache/huggingface'),
        os.path.expanduser('~/.cache/torch'),
        './cache',
        '../cache',
        os.path.expanduser('~/.cache/transformers')
    ]
    
    for cache in caches:
        if os.path.exists(cache):
            shutil.rmtree(cache)
            print(f"Cleared cache: {cache}")
    
    print("✅ System cleaned!")

def install_dependencies(extra_packages=None):
    """Install/update dependencies"""
    print("📦 Installing/updating dependencies...")
    
    base_packages = [
        "transformers", 
        "huggingface_hub", 
        "torch", 
        "torchvision",
        "torchaudio",
        "accelerate",
        "safetensors"
    ]
    
    if extra_packages:
        base_packages.extend(extra_packages)
    
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade"
    ] + base_packages)
    
    print("✅ Dependencies updated!")

def test_model(model_name, model_class=None, tokenizer_class=None):
    """Test if a model loads successfully"""
    print(f"🧪 Testing {model_name} load...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Use specific classes if provided, otherwise use Auto classes
        tokenizer_class = tokenizer_class or AutoTokenizer
        model_class = model_class or AutoModel
        
        tokenizer = tokenizer_class.from_pretrained(model_name)
        model = model_class.from_pretrained(model_name)
        
        print(f"✅ {model_name} loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ {model_name} test failed: {e}")
        return False

def get_model_preset(model_name):
    """Get model-specific configurations"""
    presets = {
        "gpt2": {
            "model_class": "GPT2LMHeadModel",
            "tokenizer_class": "GPT2Tokenizer"
        },
        "bert-base-uncased": {
            "model_class": "BertModel", 
            "tokenizer_class": "BertTokenizer"
        },
        "t5-small": {
            "model_class": "T5Model",
            "tokenizer_class": "T5Tokenizer"
        }
        # Add more presets as needed
    }
    return presets.get(model_name, {})

def main():
    parser = argparse.ArgumentParser(description='Fix Hugging Face model installation issues')
    parser.add_argument('--model', type=str, default='gpt2', 
                       help='Model name to test (default: gpt2)')
    parser.add_argument('--extra-packages', nargs='+', 
                       help='Extra packages to install')
    parser.add_argument('--clean-only', action='store_true',
                       help='Only clean system without installing/testing')
    
    args = parser.parse_args()
    
    print(f"🔧 Fixing installation for model: {args.model}")
    
    # Clean system
    clean_system(args.model)
    
    if args.clean_only:
        print("✅ Cleaning completed. Exiting.")
        return
    
    # Install dependencies
    install_dependencies(args.extra_packages)
    
    # Test the model
    success = test_model(args.model)
    
    if success:
        print(f"\n🎉 {args.model} is now working!")
    else:
        print(f"\n💡 Still having issues with {args.model}? Try:")
        print("1. Restart your runtime/kernel")
        print("2. Check model name spelling")
        print("3. Verify internet connection")
        print("4. Check Hugging Face model hub for correct model name")

if __name__ == "__main__":
    main()





### Usage Examples: ###

### Fix GPT-2 (original functionality)
#python fix_huggingface_models.py --model gpt2

### Fix BERT
#python fix_huggingface_models.py --model bert-base-uncased

### Fix custom model with extra packages
#python fix_huggingface_models.py --model microsoft/DialoGPT-medium --extra-packages datasets

### Clean only without installation
#python fix_huggingface_models.py --clean-only