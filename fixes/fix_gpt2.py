# fix_gpt2.py - Run this file separately to fix GPT-2 issues
import os
import shutil
import subprocess
import sys

def clean_system():
    print("🧹 Cleaning system for GPT-2 installation...")
    
    # Remove all potential conflict directories
    conflicts = ['gpt2', './gpt2', '../gpt2', 'models--gpt2']
    for conflict in conflicts:
        if os.path.exists(conflict):
            shutil.rmtree(conflict)
            print(f"Removed: {conflict}")
    
    # Clear caches
    caches = [
        os.path.expanduser('~/.cache/huggingface'),
        os.path.expanduser('~/.cache/torch'),
        './cache',
        '../cache'
    ]
    
    for cache in caches:
        if os.path.exists(cache):
            shutil.rmtree(cache)
            print(f"Cleared cache: {cache}")
    
    print("✅ System cleaned!")

def install_dependencies():
    print("📦 Installing/updating dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "transformers", "huggingface_hub", "torch", "torchvision"])
    print("✅ Dependencies updated!")

def test_gpt2():
    print("🧪 Testing GPT-2 load...")
    try:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        print("✅ GPT-2 loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ GPT-2 test failed: {e}")
        return False

if __name__ == "__main__":
    clean_system()
    install_dependencies()
    success = test_gpt2()
    
    if success:
        print("\n🎉 GPT-2 is now working! Return to your main notebook.")
    else:
        print("\n💡 Still having issues? Try:")
        print("1. Restart your computer")
        print("2. Run this script again")
        print("3. Check your internet connection")