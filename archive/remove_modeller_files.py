import os

if __name__ == '__main__':
    work_dir = '/cs/zbio/orzuk/projects/ContactMaps/src/Periscope'
    files = os.listdir(work_dir)
    files_to_remove = [
        os.path.join(work_dir, f) for f in files
        if f.endswith('.pdb') or f.endswith('sch') or f.endswith('D00000001')
        or f.endswith('V99990001') or f.endswith('rsr') or f.endswith('ali')
        or f.endswith('ini') or f.endswith('ent')
    ]
    for file in files_to_remove:
        os.remove(file)
