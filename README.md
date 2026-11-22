# Computer Vision Assignments

Each assignment is self-contained in its own folder, with its own README and
`requirements.txt`.

| Folder | Topic |
|---|---|
| [Assignment1](Assignment1/README.md) | Image enhancement |
| [Assignment2](Assignment2/README.md) | Image classification with transfer learning |

## Download

**Whole repository**

```bash
git clone https://github.com/EmGira/Assignments_Computer_Vision.git
cd Assignments_Computer_Vision
```

**Only one assignment** (requires git ≥ 2.25)

A sparse checkout downloads only the folder you choose, plus the files in the
repository root:

```bash
git clone --filter=blob:none --sparse https://github.com/EmGira/Assignments_Computer_Vision.git
cd Assignments_Computer_Vision
git sparse-checkout set Assignment1    # or any Assignment
```

To add another assignment later: `git sparse-checkout add Assignment2`.

## Getting started

1. Go to the folder of the assignment you want to run:

   ```bash
   cd Assignment1    # or any Assignment
   ```

2. Create a virtual environment and install the requirements of that assignment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate        # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Follow the instructions in that assignment's README to run the code.

The assignments have different dependencies, so use a separate virtual
environment for each one.
