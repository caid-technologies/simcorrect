"""Setup and environment check for Problem 5."""
import os

from paths import output_dir, smoke_test_xml_path

FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT = output_dir()

def main():
    os.makedirs(OUTPUT, exist_ok=True)
    print(f"Folder:  {FOLDER}")
    print(f"Output:  {OUTPUT}")
    files = ["render_demo.py","sim_pair.py","divergence_detector.py",
             "parameter_identifier.py","correction_and_validation.py",
             "demo.py","step0.py"]
    for f in files:
        p = os.path.join(FOLDER, f)
        print(f"  {f}: {'OK' if os.path.exists(p) else 'MISSING'}")
    opencad_path = os.path.join(os.path.dirname(FOLDER), "opencad.py")
    print(f"  opencad.py (root): {'OK' if os.path.exists(opencad_path) else 'MISSING'}")
    try:
        from opencad import Part
        p = Part("grip").set_mass(0.160)
        p.export(str(smoke_test_xml_path()))
        print("  OpenCAD import:    OK")
    except Exception as e:
        print(f"  OpenCAD import:    FAILED -- {e}")
    try:
        import mujoco; print(f"  mujoco:            {mujoco.__version__}  OK")
    except ImportError: print("  mujoco:            NOT FOUND")
    try:
        import imageio; print("  imageio:           OK")
    except ImportError: print("  imageio:           NOT FOUND")
    try:
        from PIL import Image; print("  Pillow:            OK")
    except ImportError: print("  Pillow:            NOT FOUND")
    print("\nStep 0 complete. Run order:")
    print("  uv run --project .. python demo.py")
    print("  uv run --project .. python sim_pair.py")
    print("  uv run --project .. python correction_and_validation.py")
    print("  uv run --project .. python render_demo.py")

if __name__ == "__main__":
    main()
