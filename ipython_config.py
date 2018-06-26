#######
# Please place this file into: ~/.ipython/profile_default/ipython_config.py
# In order to make use of it
#######

### If you want to auto-save .html and .py versions of your notebook:
# modified from: https://github.com/ipython/ipython/issues/8009
import os
from subprocess import check_call

def post_save(model, os_path, contents_manager):
    """post-save hook for converting notebooks to .py scripts"""
    if model['type'] != 'notebook':
        return # only do this for notebooks
    d, fname = os.path.split(os_path)
    check_call(['ipython', 'nbconvert', '--to', 'script', fname], cwd=d)
    check_call(['ipython', 'nbconvert', '--to', 'html', fname], cwd=d)

c = get_config()
c.FileContentsManager.post_save_hook = post_save

# Run all nodes interactively
c.InteractiveShell.ast_node_interactivity = "all"
