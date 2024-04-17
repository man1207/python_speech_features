import matplotlib.pyplot as plt

def visualize_wave(signal, sample_rate, loss_label=None, start_ms=None, end_ms=None, loss_color='red', frame_size=80):

    # Convert start and end times in ms to sample indices
    if start_ms is not None:
        start_idx = int(start_ms * sample_rate // 1000)
    else:
        start_idx = 0

    if end_ms is not None:
        end_idx = int(end_ms * sample_rate // 1000)
    else:
        end_idx = len(data)

    # Adjust start and end indices to be multiples of frame_size
    start_idx = start_idx - (start_idx % frame_size)
    end_idx = end_idx - (end_idx % frame_size)

    # Extract the portion of the data to visualize
    data = data[start_idx:end_idx]

    plt.figure(figsize=(20, 4))
    if loss_label is not None:
        # Separate the data into lossless and lossy segments
        data_lossless = data.copy()
        data_lossy = data.copy()
        data_lossy = data_lossy.astype(float)
        data_lossless = data_lossless.astype(float)
        # Correct the indices calculation
        for i in range((end_idx - start_idx) // frame_size):
            if loss_label[i + start_idx // frame_size]:
                data_lossless[i * frame_size : (i + 1) * frame_size] = np.nan  # invalidate lossy data in lossless segments
            else:
                data_lossy[i * frame_size : (i + 1) * frame_size] = np.nan
                # Plot the lossless data
        plt.plot(data_lossless, color='blue')

        # Overlay the lossy data
        plt.plot(data_lossy, color=loss_color)
    else:
        plt.plot(data, color='blue')

    plt.title("Форма сигнала")
    plt.xlabel("Сэмпл")
    plt.ylabel("Амплитуда")

    plt.xticks(np.arange(0, end_idx - start_idx, frame_size))
    plt.show()
