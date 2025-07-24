"""
Basic usage example for Dynamic SSM Transformer
"""

from transformers import AutoTokenizer
import torch
import sys
sys.path.append('../src')

from dynamic_ssm.models.hybrid_model import CompleteAdaptiveHybridModel

def main():
    # Initialize model
    print("🚀 Loading Dynamic SSM Transformer...")
    
    model = CompleteAdaptiveHybridModel(
        base_model_name="google/gemma-2b",  # or your preferred model, I used for this model the gemma 3n series
        num_hybrid_layers=4,
        gate_bias=-1.0,  # Moderate SSM usage
        memory_size=256,
        use_tools=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Example 1: Simple generation
    print("\n📝 Example 1: Simple Generation")
    prompt = "The capital of France is"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=50,
        temperature=0.7,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    
    # Example 2: Generation with tools
    print("\n📝 Example 2: Generation with Tools")
    prompt = "What is the square root of 625?"
    
    result = model.generate_with_tools(
        prompt,
        tokenizer,
        max_length=50,
        temperature=0.7
    )
    
    print(f"Prompt: {prompt}")
    print(f"Response: {result['text']}")
    
    if result['tool_responses']:
        print("\n🔧 Tools used:")
        for step, tool, response in result['tool_responses']:
            print(f"  - {tool}: {response}")
    
    # Example 3: Check gate statistics
    print("\n📊 Gate Statistics:")
    stats = model.get_gate_statistics()
    for layer, data in stats.items():
        print(f"  {layer}: mean={data['mean']:.3f}, active={data['mean'] > 0.5}")
    
    # Example 4: Multi-turn conversation
    print("\n📝 Example 3: Multi-turn Conversation")
    conversation = [
        "Tell me about machine learning",
        "How does it relate to AI?",
        "What are some practical applications?"
    ]
    
    context = ""
    for turn in conversation:
        full_prompt = context + "\nUser: " + turn + "\nAssistant:"
        
        result = model.generate_with_tools(
            full_prompt,
            tokenizer,
            max_length=100,
            temperature=0.7
        )
        
        response = result['text'].split("Assistant:")[-1].strip()
        print(f"\nUser: {turn}")
        print(f"Assistant: {response}")
        
        context = full_prompt + " " + response

if __name__ == "__main__":
    main()
