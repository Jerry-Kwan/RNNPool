def get_train_loss(file_path):
    """Get train loss from log file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        loss = []
        
        for line in lines:
            if line[:10] == 'train_loss':
                loss.append(float(line.split()[1]))

        return loss


def get_train_acc(file_path):
    """Get train acc from log file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        acc = []
        
        for line in lines:
            if line[:10] == 'train_loss':
                acc.append(float(line.split()[3]))

        return acc


def get_test_loss(file_path):
    """Get test loss from log file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        loss = []
        
        for line in lines:
            if line[:9] == 'test_loss':
                loss.append(float(line.split()[1]))

        return loss


def get_test_acc(file_path):
    """Get test acc from log file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        acc = []

        for line in lines:
            if line[:9] == 'test_loss':
                acc.append(float(line.split()[3]))

        return acc

def get_epoch_time(file_path):
    """Get training time per epoch from log file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        epoch_time = []
        
        for line in lines:
            if line[:5] == 'total':
                epoch_time.append(float(line.split()[5]))
        
        return epoch_time
