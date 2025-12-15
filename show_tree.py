# show_tree_v2.py
import os
import fnmatch

# --- CONFIGURATION ---------------------------------------------------
# Set the maximum depth to display. 3 is a good starting point.
MAX_DEPTH = 3

# Set to False to only show directories and not the files inside them.
SHOW_FILES = True

# Add names of directories, files, or extensions to ignore.
# Uses wildcard matching (e.g., '*.pyc' matches all .pyc files).
IGNORE_LIST = {
    # Common dev folders
    '.git', '__pycache__', 'venv', '.vscode', '.idea', 'node_modules',
    
    # Common build/dist folders
    'build', 'dist', 'target',
    
    # Common data and media file extensions
    '*.json', '*.csv', '*.png', '*.jpg', '*.jpeg', '*.gif', '*.svg',
    '*.db', '*.sqlite3',
    
    # Radiance specific (based on your output)
    '*.oct', '*.rad',
    
    # Other common noise
    '.DS_Store', '*.pyc', '*.log', 'pyvenv.cfg'
}
# ---------------------------------------------------------------------

def generate_project_tree():
    """
    Generates and prints a tree structure of the project directory
    with depth limit and advanced ignoring.
    """
    output_lines = []
    
    def is_ignored(name):
        """Check if a file or directory should be ignored."""
        for pattern in IGNORE_LIST:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False

    def build_tree(dir_path, prefix="", depth=0):
        """Recursively builds the directory tree."""
        if depth > MAX_DEPTH:
            return
            
        try:
            # Get items and filter out ignored ones
            items = [item for item in os.listdir(dir_path) if not is_ignored(item)]
            
            # Separate directories and files
            dirs = sorted([d for d in items if os.path.isdir(os.path.join(dir_path, d))])
            files = sorted([f for f in items if not os.path.isdir(os.path.join(dir_path, f))])

            if SHOW_FILES:
                entries = dirs + files
            else:
                entries = dirs
            
            pointers = ['├── '] * (len(entries) - 1) + ['└── ']
            
            for pointer, entry in zip(pointers, entries):
                output_lines.append(f"{prefix}{pointer}{entry}")
                path = os.path.join(dir_path, entry)
                if os.path.isdir(path):
                    extension = '│   ' if pointer == '├── ' else '    '
                    build_tree(path, prefix + extension, depth + 1)

        except PermissionError:
            output_lines.append(f"{prefix}└── [Permission Denied]")
            return

    project_root = os.getcwd()
    project_name = os.path.basename(project_root)
    output_lines.append(f"{project_name}/")
    build_tree(project_root, depth=0)

    return "\n".join(output_lines)

if __name__ == "__main__":
    print("--------------------------------------------------")
    print("Project Directory Structure:")
    print(f"(Max Depth: {MAX_DEPTH}, Show Files: {SHOW_FILES})")
    print("--------------------------------------------------")
    
    project_structure = generate_project_tree()
    print(project_structure)
    
    print("\n--------------------------------------------------")
    print("✅ Done! Copy the text above this line.")
    print("💡 To change depth or ignore more files, edit this script.")
    print("--------------------------------------------------")
