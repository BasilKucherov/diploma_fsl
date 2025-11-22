import sys
import os

# Add current dir to path to find datasets module
sys.path.append(os.path.join(os.path.dirname(__file__), 'fsl', 'cdfsl'))

print("Verifying EuroSAT...")
try:
    from fsl.cdfsl.datasets import EuroSAT_few_shot
    tl = EuroSAT_few_shot.TransformLoader(224)
    # This should FAIL if not patched, because "Scale" is not in transforms
    # But due to our fix, it should handle "Scale" without calling getattr(transforms, "Scale")
    try:
        t = tl.parse_transform("Scale")
        print("  SUCCESS: parse_transform('Scale') returned:", t)
    except AttributeError as e:
        print("  FAILURE: parse_transform('Scale') raised AttributeError:", e)
except Exception as e:
    print(f"  ERROR importing/initializing: {e}")

print("\nVerifying ISIC...")
try:
    from fsl.cdfsl.datasets import ISIC_few_shot
    tl = ISIC_few_shot.TransformLoader(224)
    try:
        t = tl.parse_transform("Scale")
        print("  SUCCESS: parse_transform('Scale') returned:", t)
    except AttributeError as e:
        print("  FAILURE: parse_transform('Scale') raised AttributeError:", e)
except Exception as e:
    print(f"  ERROR importing/initializing: {e}")

print("\nVerifying ChestX...")
try:
    from fsl.cdfsl.datasets import Chest_few_shot
    tl = Chest_few_shot.TransformLoader(224)
    try:
        t = tl.parse_transform("Scale")
        print("  SUCCESS: parse_transform('Scale') returned:", t)
    except AttributeError as e:
        print("  FAILURE: parse_transform('Scale') raised AttributeError:", e)
except Exception as e:
    print(f"  ERROR importing/initializing: {e}")

