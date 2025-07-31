#!/usr/bin/env python3
"""
Test script to validate wandb integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffsynth.trainers.utils import ModelLogger, WANDB_AVAILABLE

def test_wandb_integration():
    """Test basic wandb integration functionality"""
    
    print("Testing wandb integration...")
    print(f"Wandb available: {WANDB_AVAILABLE}")
    
    if not WANDB_AVAILABLE:
        print("❌ wandb is not installed. Install with: pip install wandb")
        return False
    
    # Test ModelLogger with wandb enabled
    try:
        logger = ModelLogger(
            output_path="./test_output",
            use_wandb=True
        )
        print("✅ ModelLogger with wandb created successfully")
        
        # Test step logging (without actual wandb.init)
        logger.step_count = 0
        logger.use_wandb = False  # Disable actual logging for test
        logger.on_step_end(0.5)  # Test with float loss
        
        import torch
        logger.on_step_end(torch.tensor(0.3))  # Test with tensor loss
        
        print("✅ Step logging test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing wandb integration: {e}")
        return False

def test_parser_args():
    """Test that wandb arguments are properly added to parser"""
    
    from diffsynth.trainers.utils import wan_parser
    
    parser = wan_parser()
    
    # Test parsing with wandb arguments
    test_args = [
        "--dataset_base_path", "/test/path",
        "--use_wandb",
        "--wandb_project", "test-project",
        "--wandb_run_name", "test-run",
        "--wandb_entity", "test-entity"
    ]
    
    try:
        args = parser.parse_args(test_args)
        
        assert args.use_wandb == True
        assert args.wandb_project == "test-project"
        assert args.wandb_run_name == "test-run"
        assert args.wandb_entity == "test-entity"
        
        print("✅ Parser arguments test passed")
        return True
        
    except Exception as e:
        print(f"❌ Error testing parser arguments: {e}")
        return False

if __name__ == "__main__":
    print("Running wandb integration tests...\n")
    
    success = True
    success &= test_wandb_integration()
    success &= test_parser_args()
    
    if success:
        print("\n🎉 All tests passed!")
        print("\nTo use wandb logging in training:")
        print("1. Install wandb: pip install wandb")
        print("2. Login: wandb login")
        print("3. Add --use_wandb flag to your training command")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)
