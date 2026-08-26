

#Updated to include all stuff

import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import tempfile
import sys

# GUI Setup 
root = tk.Tk()
root.title("BTR Analysis Tool")
root.geometry("650x650")

# Inputting

tk.Label(root, text="Working Directory:").pack(pady=3)
folder_entry = tk.Entry(root, width=60)
folder_entry.pack()

#For Directory Choosing
def browse_folder():
    folder = filedialog.askdirectory()
    if folder:
        folder_entry.delete(0, tk.END)
        folder_entry.insert(0, folder)


tk.Button(root, text="Browse Folder or Directory", command=browse_folder).pack(pady=5)

#For typing the env (usually fly)
tk.Label(root, text="Conda Environment Name:").pack(pady=3)
env_entry = tk.Entry(root, width=60)
env_entry.pack()

#For Choosing Config file
tk.Label(root, text="Path to Config File (.toml):").pack(pady=3)
file_entry = tk.Entry(root, width=60)
file_entry.pack()


def browse_file():
    file = filedialog.askopenfilename()
    if file:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file)


tk.Button(root, text="Browse Config File", command=browse_file).pack(pady=5)

#For choosing the Cleaner file based on the apparatus
tk.Label(root, text="Path to the Rebin for Cleaner:").pack(pady=3)
file2_entry = tk.Entry(root, width=60)
file2_entry.pack()

rebin_path = r"C:\Users\HamadaLab2\OneDrive\2023_project_hamada_fly\Rebin.py"

def browse_file2():
    file = filedialog.askopenfilename()
    if file:
        file2_entry.delete(0, tk.END)
        file2_entry.insert(0, file)


tk.Button(root, text="Browse Cleaner File", command=browse_file2).pack(pady=5)

tk.Label(root, text="Path to predictions.csv file:").pack(pady=3)
file3_entry = tk.Entry(root, width=60)
file3_entry.pack()

def browse_file3():
    file = filedialog.askopenfilename()
    if file:
        file3_entry.delete(0, tk.END)
        file3_entry.insert(0, file)


tk.Button(root, text="Browse Predictions File", command=browse_file3).pack(pady=5)

# --- Command sets now support multiple commands ---
command_set_1 = [
    {
        "label": "Register and Detect",
        "commands": [
            'python -m src register "%FILE%"',
            'python -m src detect "%FILE%"'
        ]
    },
    {
        "label": "placeholder",
        "commands": [
            'echo File is: "%FILE%"',
            'dir'
        ]
    }
]

command_set_2 = [
    {
        "label": "Cleaner",
        "commands": [
            'python "%FILE2%" "%FILE3%" "hansolo"',
            'echo File2 processed.'
        ]
    },
    {
        "label": "placeholder",
        "commands": [
            'echo Secondary file is "%FILE2%"',
            'timeout /t 2'
        ]
    }
]


# --- Updated function to handle multiple commands ---
def run_command_in_conda_env(command_list, use_file2=False, use_file3=False):
    folder = folder_entry.get().strip()
    env = env_entry.get().strip()
    file_path = file_entry.get().strip()
    file2_path = file2_entry.get().strip()
    file3_path = file3_entry.get().strip()

    if not folder or not os.path.isdir(folder):
        messagebox.showerror("Error", "Please select a valid working directory.")
        return
    if not env:
        messagebox.showerror("Error", "Please enter the Conda environment name.")
        return
    if not file_path or not os.path.isfile(file_path):
        messagebox.showerror("Error", "Please select a valid primary file path.")
        return
    if use_file2 and (not file2_path or not os.path.isfile(file2_path)):
        messagebox.showerror("Error", "Please select a valid secondary file path.")
        return
    if use_file3 and (not file3_path or not os.path.isfile(file3_path)):
        messagebox.showerror("Error", "Please select a valid tertiary file path.")
        return
    
    # Build the batch lines
    batch_lines = [
        "@echo off",
        f"call conda activate {env}",
        f"set FILE=\"{file_path}\""
    ]
    if use_file2:
        batch_lines.append(f"set FILE2=\"{file2_path}\"")
    
    if use_file3:
        batch_lines.append(f"set FILE2=\"{file2_path}\"")

    batch_lines.append(f"cd /d \"{folder}\"")

    # Replace placeholders in each command and add to batch
    for cmd in command_list:
        cmd_replaced = cmd.replace("%FILE%", f'"{file_path}"').replace("%FILE2%", f'"{file2_path}"').replace("%FILE3%", f'"{file3_path}"')
        batch_lines.append(cmd_replaced)

    batch_lines.append("pause")

    # Write and run the batch file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bat", mode='w', encoding='utf-8') as bat_file:
            bat_file.write("\n".join(batch_lines))
            bat_path = bat_file.name

        subprocess.Popen(["cmd.exe", "/k", bat_path], shell=True)

    except Exception as e:
        messagebox.showerror("Execution Error", str(e))
    
    {"label": "Echo File 2", "command": 'echo Secondary file is "%FILE2%"'},



# Buttons for %FILE%
tk.Label(root, text="Commands using Config File:").pack(pady=8)

for cmd in command_set_1:
    tk.Button(
        root,
        text=cmd["label"],
        width=40,
        command=lambda c=cmd["commands"]: run_command_in_conda_env(c, use_file2=False, use_file3=False)
    ).pack(pady=3)

# Buttons for %FILE2%
tk.Label(root, text="Commands using Rebin and Predictions:").pack(pady=8)

for cmd in command_set_2:
    tk.Button(
        root,
        text=cmd["label"],
        width=40,
        bg="#f0f0d0",
        command=lambda c=cmd["commands"]: run_command_in_conda_env(c, use_file2=True, use_file3=True)
    ).pack(pady=3)


root.mainloop()
