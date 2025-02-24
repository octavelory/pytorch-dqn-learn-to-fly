import modal
from learn import train_dqn, PlaneEnv

app = modal.App()
ai_storage = modal.Volume.from_name("ai_storage", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch")
    .pip_install("pybullet")
    .pip_install("numpy")
    .pip_install("gym")
    .pip_install("matplotlib")
    .pip_install("pandas")
)

@app.function(gpu="T4", image=image, volumes={"/data": ai_storage})
def run():
    env = PlaneEnv(gui=False)
    train_dqn(env, episodes=1000)

@app.local_entrypoint()
def main():
    run.remote()