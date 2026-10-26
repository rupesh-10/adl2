from train_mnist import run_experiment

if __name__ == '__main__':
    run_experiment(name="Standard UNet AE", denoising=False, epochs=5)
    run_experiment(name="Denoising UNet AE", denoising=True, epochs=5)
