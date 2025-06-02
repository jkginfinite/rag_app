import ast
import pkg_resources
import sys
import os

def extract_imports_from_file(file_path):
    with open(file_path, "r") as f:
        tree = ast.parse(f.read())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return sorted(imports)

def get_installed_version(package_name):
    try:
        return pkg_resources.get_distribution(package_name).version
    except Exception:
        return None

def generate_requirements(script_path, output_file="requirements_gen.txt"):
    print(f"Scanning imports in {script_path}")
    imports = extract_imports_from_file(script_path)
    with open(output_file, "w") as f:
        for package in imports:
            version = get_installed_version(package)
            if version:
                f.write(f"{package}=={version}\n")
            else:
                print(f"⚠️  Package '{package}' not found in current environment.")
    print(f"✅ requirements.txt created with {len(imports)} packages.")

# Example usage:
# Replace 'your_script.py' with the script you want to analyze
if __name__ == "__main__":
    script_file = sys.argv[1] if len(sys.argv) > 1 else "your_script.py"
    if not os.path.exists(script_file):
        print(f"❌ File '{script_file}' not found.")
    else:
        generate_requirements(script_file)
