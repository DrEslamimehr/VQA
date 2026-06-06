# Git Push Instructions

Create a new empty remote repository, then run:

```bash
git remote add origin <new-repo-url>
git branch -M main
git add .
git commit -m "Reconstruct QA-TM reference implementation"
git push -u origin main
```

If a remote already exists:

```bash
git remote set-url origin <new-repo-url>
git branch -M main
git push -u origin main
```

