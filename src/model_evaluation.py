def evaluate_model(model, dataset):
    loss, accuracy = model.evaluate(dataset)
    print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.2f}')
