# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import json

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

from braket.jobs import load_job_checkpoint, save_job_checkpoint, save_job_result
from braket.jobs.metrics import log_metric

from var_classifier import utils

N_QUBITS = 2

def statepreparation(a):
    qml.RY(a[0], wires=0)

    qml.CNOT(wires=[0, 1])
    qml.RY(a[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[2], wires=1)

    qml.PauliX(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[3], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[4], wires=1)
    qml.PauliX(wires=0)


def layer(W):
    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
    qml.CNOT(wires=[0, 1])

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def circuit(weights, angles):
    statepreparation(angles)
    
    for W in weights:
        layer(W)

    return qml.expval(qml.PauliZ(0))


def variational_classifier(weights, bias, angles):
    return circuit(weights, angles) + bias


def cost(weights, bias, features, labels):
    predictions = [variational_classifier(weights, bias, f) for f in features]
    return utils.square_loss(labels, predictions)
    
def main(input_dir, hp_file, device_arn):
    # Read the hyperparameters
    with open(hp_file, "r") as f:
        hyperparams = json.load(f)
    print(hyperparams)
    print("\n")

    seed = int(hyperparams["seed"])
    num_iterations = int(hyperparams["num_iterations"])
    stepsize = float(hyperparams["stepsize"])
    batchsize = int(hyperparams["batchsize"])
    num_layers = int(hyperparams["num_layers"])


    if "copy_checkpoints_from_job" in hyperparams:
        copy_checkpoints_from_job = hyperparams["copy_checkpoints_from_job"].split(
            "/", 2
        )[-1]
    else:
        copy_checkpoints_from_job = None

    feats_train = np.loadtxt(f"{input_dir}/input/irisdata_train_features.txt")
    y_train = np.loadtxt(f"{input_dir}/input/irisdata_train_labels.txt")

    feats_val = np.loadtxt(f"{input_dir}/input/irisdata_val_features.txt")
    y_val = np.loadtxt(f"{input_dir}/input/irisdata_val_labels.txt")

    num_train = feats_train.shape[0]

    np.random.seed(
        seed
    )
    opt = NesterovMomentumOptimizer(stepsize=stepsize)

    # Load checkpoint if it exists
    if copy_checkpoints_from_job:
        checkpoint_1 = load_job_checkpoint(
            copy_checkpoints_from_job,
            checkpoint_file_suffix="checkpoint-1",
        )
        start_iteration = checkpoint_1["iteration"]
        weights = qml.numpy.array(checkpoint_1["weights"], requires_grad=True)
        bias = qml.numpy.array(checkpoint_1["bias"], requires_grad=True)
        print("Checkpoint loaded")
    else:
        start_iteration = 0
        weights = 0.01 * np.random.randn(num_layers, N_QUBITS, 3, requires_grad=True)
        bias = np.array(0.0, requires_grad=True)

    for it in range(start_iteration, num_iterations):

        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, num_train, (batchsize,))
        feats_train_batch = feats_train[batch_index]
        y_train_batch = y_train[batch_index]
        weights, bias, _, _ = opt.step(
            cost, weights, bias, feats_train_batch, y_train_batch
        )

        # Compute predictions on train and validation set
        predictions_train = [
            np.sign(variational_classifier(weights, bias, f)) for f in feats_train
        ]
        predictions_val = [
            np.sign(variational_classifier(weights, bias, f)) for f in feats_val
        ]

        # Compute metrics on train and validation set
        acc_train = utils.accuracy(y_train, predictions_train)
        acc_val = utils.accuracy(y_val, predictions_val)

        cost_train = cost(weights, bias, feats_train, y_train)
        cost_val = cost(weights, bias, feats_val, y_val)

        # Log metrics before the update
        log_metric(
            metric_name="acc_train",
            value=acc_train,
            iteration_number=it,
        )

        log_metric(
            metric_name="acc_val",
            value=acc_val,
            iteration_number=it,
        )

        log_metric(
            metric_name="cost_train",
            value=cost_train,
            iteration_number=it,
        )

        log_metric(
            metric_name="cost_val",
            value=cost_val,
            iteration_number=it,
        )

        # Save the current params and previous cost to a checkpoint
        save_job_checkpoint(
            checkpoint_data={
                "iteration": it,
                "weights": weights.numpy().tolist(),
                "bias": bias.numpy(),
                "cost_train": cost_train.numpy(),
            },
            checkpoint_file_suffix="checkpoint-1",
        )

    # Save the final result
    save_job_result(
        {
            "weights": weights.numpy().tolist(),
            "bias": bias.numpy(),
            "final_cost_train": cost_train.numpy(),
            "final_cost_val": cost_val.numpy(),
            "final_acc_train": acc_train,
            "final_acc_val": acc_val,
        }
    )


if __name__ == "__main__":
    input_dir = os.environ["AMZN_BRAKET_INPUT_DIR"]
    hp_file = os.environ["AMZN_BRAKET_HP_FILE"]
    device_arn = os.environ["AMZN_BRAKET_DEVICE_ARN"]

    main(input_dir=input_dir, hp_file=hp_file, device_arn=device_arn)
