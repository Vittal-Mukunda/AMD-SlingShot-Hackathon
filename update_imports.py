import os, glob, re

root = r'd:\Projects\amd\AMD-SlingShot-Hackathon'
for ext in ('**/*.py',):
    for path in glob.glob(os.path.join(root, ext), recursive=True):
        if 'venv' in path or '.venv' in path: continue
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        modified = content
        modified = re.sub(r'(?m)^.*?sys\.path\.append.*$\n?', '', modified)
        
        # Replace import config -> from slingshot.core.settings import config
        modified = re.sub(r'(?m)^import config$', 'from slingshot.core.settings import config', modified)
        
        # Replace import config as cfg -> from slingshot.core.settings import config as cfg
        modified = re.sub(r'(?m)^(\s*)import config as (.*)$', r'\1from slingshot.core.settings import config as \2', modified)
        
        # Replace from environment..., from agents..., etc.
        packages = ['environment', 'agents', 'training', 'evaluation', 'visualization', 'baselines']
        for pkg in packages:
            modified = re.sub(fr'(?m)^(\s*)from {pkg}(\.)', fr'\1from slingshot.{pkg}\2', modified)
            modified = re.sub(fr'(?m)^(\s*)from {pkg} ', fr'\1from slingshot.{pkg} ', modified)
            modified = re.sub(fr'(?m)^(\s*)import {pkg}', fr'\1import slingshot.{pkg}', modified)

        if modified != content:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(modified)
