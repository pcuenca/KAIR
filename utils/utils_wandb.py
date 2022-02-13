import glob
import os
import shutil
import wandb

# W&B helpers

def wandb_enabled(opt):
    return 'wandb' in opt

def wandb_init(opt):
    if not wandb_enabled(opt):
        return

    entity = opt['wandb'].get('entity', None)
    project = opt['wandb'].get('project', None)
    tags = opt['wandb'].get('tags', None)
    save_code = opt['wandb'].get('save_code', False)

    wandb.init(
        project=project,
        entity=entity,
        tags=tags,
        config=opt,
        save_code=save_code,
    )

# Maybe move to model class
def log_artifact(opt, model, current_step):
    if not wandb_enabled(opt):
        return
    # Prepend run id to artifact name so it is attributed to the current run
    name = opt['task']
    name = f'{wandb.run.id}_{name}'
    artifact = wandb.Artifact(name, type='model')
    for path in glob.glob(os.path.join(model.save_dir, f'{current_step}_*')):
        artifact.add_file(path)
    wandb.run.log_artifact(artifact)

def log_to_wandb(opt, logs):
    if not wandb_enabled(opt):
        return
    wandb.log(logs)

def download_wandb_checkpoints(opt):
    """If pretrained_netG looks like a wandb artifact, download it and copy files to models path."""
    pretrained_netG = opt['path']['pretrained_netG']
    if pretrained_netG is not None and ":" in pretrained_netG:
        print("Downloading checkpoint from wandb artifact")
        artifact = wandb.Api().artifact(pretrained_netG)
        artifact_path = artifact.download()
        for file in os.listdir(artifact_path):
            shutil.copy(os.path.join(artifact_path, file), opt['path']['models'])
