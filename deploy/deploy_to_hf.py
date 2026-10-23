"""
Publish the demo app as a free Hugging Face Space - run this on YOUR computer.

One-time setup:
    1. Make a free account at https://huggingface.co/join
    2. Create a token with WRITE access:
       https://huggingface.co/settings/tokens  ->  New token  ->  Write
    3. pip install huggingface_hub

Deploy (from the repo folder):
    python deploy/deploy_to_hf.py --space YOUR_HF_USERNAME/tomato-leaf-doctor

The script asks for the token (or set HF_TOKEN, or run `hf auth login`
once), uploads only the files the app needs (~20 MB), and prints the
Space URL. The first build takes ~5 minutes; after that the link works
for anyone, on any device. Run it again to push updates.

Check what would be uploaded without uploading anything:
    python deploy/deploy_to_hf.py --space YOUR_HF_USERNAME/tomato-leaf-doctor --dry-run
"""

import argparse
import getpass
import os
import shutil
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Space path -> source path in this repo
FILES = {
    "app.py": "app.py",
    "train.py": "train.py",  # load_trained_model() / build_mobilenet()
    "gradcam.py": "gradcam.py",  # heatmap shown next to each prediction
    "README.md": "deploy/space_README.md",
    "requirements.txt": "deploy/space_requirements.txt",
    "outputs/plant_disease_subset.keras": "outputs/plant_disease_subset.keras",
    "outputs/plant_disease_subset.weights.h5": "outputs/plant_disease_subset.weights.h5",
    "outputs/class_names.json": "outputs/class_names.json",
    "outputs/leaf_check.json": "outputs/leaf_check.json",  # non-leaf rejection
    # figures for the "How it works" tab
    "outputs/confusion_matrix_test.png": "outputs/confusion_matrix_test.png",
    "outputs/training_curves.png": "outputs/training_curves.png",
    "outputs/gradcam_grid.png": "outputs/gradcam_grid.png",
}


def stage(dst: Path) -> list:
    """Copy the Space's files into dst; return their Space paths."""
    staged = []
    for space_path, src in FILES.items():
        src = REPO / src
        if not src.exists():
            raise SystemExit(f"Missing {src} - run this from an up-to-date clone "
                             "(git pull), or train first so outputs/ exists.")
        (dst / space_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst / space_path)
        staged.append(space_path)
    for img in sorted((REPO / "samples").glob("*.jpg")):  # app examples
        (dst / "samples").mkdir(exist_ok=True)
        shutil.copy2(img, dst / "samples" / img.name)
        staged.append(f"samples/{img.name}")
    return staged


def get_token(cli_token):
    if cli_token:
        return cli_token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    from huggingface_hub import get_token as cached_token
    token = cached_token()  # from an earlier `hf auth login`
    if token:
        return token
    return getpass.getpass("Hugging Face token (write access, input hidden): ").strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True,
                        help="USERNAME/space-name, e.g. ashwanth/tomato-leaf-doctor")
    parser.add_argument("--token", help="HF write token (default: HF_TOKEN or saved login)")
    parser.add_argument("--private", action="store_true",
                        help="Make the Space private (only you can open it)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only list the files that would be uploaded")
    args = parser.parse_args()
    if args.space.count("/") != 1:
        raise SystemExit("--space must look like USERNAME/space-name")

    with tempfile.TemporaryDirectory() as tmp:
        staged = stage(Path(tmp))
        size = sum(f.stat().st_size for f in Path(tmp).rglob("*") if f.is_file())
        print(f"{len(staged)} files, {size / 1e6:.1f} MB:")
        for p in staged:
            print("  ", p)
        if args.dry_run:
            print("\nDry run - nothing uploaded.")
            return

        from huggingface_hub import HfApi
        api = HfApi(token=get_token(args.token))
        user = api.whoami()["name"]
        print(f"\nLogged in as {user}")
        api.create_repo(args.space, repo_type="space", space_sdk="gradio",
                        private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.space, repo_type="space", folder_path=tmp,
                          commit_message="Deploy tomato leaf disease demo")

    url = f"https://huggingface.co/spaces/{args.space}"
    print(f"\nUploaded. The Space is building (about 5 minutes the first time):\n  {url}")
    print("Watch progress in the 'Logs' tab there; when it says Running, the app is live.")


if __name__ == "__main__":
    main()
