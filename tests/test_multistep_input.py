import torch


def fast_construct_train(x, y, pred_range, history, interval):
    inputs = x.unsqueeze(0).repeat_interleave(history, dim=0)
    for t in range(history):
        inputs[t] = inputs[t].roll(-t * interval, dims=0)

    last_idx = -((history - 1) * interval + pred_range)

    outputs = y.roll(last_idx, dims=0)

    inputs = inputs[:, :last_idx].transpose(0, 1)
    outputs = outputs[:last_idx]

    return inputs, outputs


def slow_construct_train(x, y, pred_range, history, interval):
    inputs = []
    outputs = []
    for i in range(x.shape[0]):
        if i + (history - 1) * interval + pred_range >= x.shape[0]:
            break
        inputs.append([x[i + t * interval] for t in range(history)])
        inputs[i] = torch.stack(inputs[i], dim=0)
        outputs.append(y[i + (history - 1) * interval + pred_range])
    inputs = torch.stack(inputs, dim=0)
    outputs = torch.stack(outputs, dim=0)
    return inputs, outputs


def old_construct_train(x, y, pred_range):  # equivalent to history=1
    inputs = x[0:-pred_range].unsqueeze(1)
    outputs = y[pred_range:]
    return inputs, outputs


def test_train_1():
    pred_range = 72
    history = 3
    interval = 6
    x, y = torch.randn((730, 5, 32, 64)), torch.randn((730, 3, 32, 64))
    in1, out1 = fast_construct_train(x, y, pred_range, history, interval)
    in2, out2 = slow_construct_train(x, y, pred_range, history, interval)
    return torch.all(in1 == in2) & torch.all(out1 == out2)


def test_train_2():
    pred_range = 72
    history = 1
    interval = 0
    x, y = torch.randn((730, 5, 32, 64)), torch.randn((730, 3, 32, 64))
    in1, out1 = fast_construct_train(x, y, pred_range, history, interval)
    in2, out2 = old_construct_train(x, y, pred_range)
    return torch.all(in1 == in2) & torch.all(out1 == out2)


print(test_train_1())
print(test_train_2())


def fast_construct_val(x, y, pred_range, history, interval, pred_steps):
    inputs = x.unsqueeze(0).repeat_interleave(history, dim=0)
    for t in range(history):
        inputs[t] = inputs[t].roll(-t * interval, dims=0)

    outputs = y.unsqueeze(0).repeat_interleave(pred_steps, dim=0)
    start_idx = (history - 1) * interval + pred_range
    for t in range(pred_steps):
        outputs[t] = outputs[t].roll(-(start_idx + t * pred_range), dims=0)

    last_idx = -((history - 1) * interval + pred_steps * pred_range)

    inputs = inputs[:, :last_idx].transpose(0, 1)
    outputs = outputs[:, :last_idx].transpose(0, 1)

    return inputs, outputs


def old_construct_val(x, y, pred_range, pred_steps):
    inputs = []
    outputs = []
    in_channels = x.shape[1]
    out_channels = y.shape[1]
    for k in range(in_channels):
        interval = pred_range * pred_steps
        inputs.append(x[:, k][0:-interval])

        if k < out_channels:
            output_k = []
            for step in range(pred_steps):
                start = (step + 1) * pred_range
                end = (step - pred_steps + 1) * pred_range if step != pred_steps - 1 else x.shape[0]
                output_k.append(y[:, k][start:end])

            output_k = torch.stack(output_k, dim=1)
            outputs.append(output_k)

    inputs = torch.stack(inputs, dim=1).unsqueeze(1)
    outputs = torch.stack(outputs, dim=2)
    return inputs, outputs


def test_val():
    pred_range = 72
    history = 1
    interval = 0
    pred_steps = 1
    x, y = torch.randn((730, 5, 32, 64)), torch.randn((730, 3, 32, 64))
    in1, out1 = fast_construct_val(x, y, pred_range, history, interval, pred_steps)
    in2, out2 = old_construct_val(x, y, pred_range, pred_steps)
    return torch.all(in1 == in2) & torch.all(out1 == out2)


print(test_val())
