import torchvision.utils as vutils

def generate_with_A(inputs, path, G_AB, G_BA, idx=None):
    x_AB = G_AB(inputs)
    x_ABA = G_BA(x_AB)

    input_path = '{}/{}_input_A.png'.format(path, idx)
    x_AB_path = '{}/{}_x_AB.png'.format(path, idx)
    x_ABA_path = '{}/{}_x_ABA.png'.format(path, idx)
    vutils.save_image(inputs, input_path, normalize=True)
    vutils.save_image(x_AB.data, x_AB_path, normalize=True)
    print("[*] Samples saved: {}".format(x_AB_path))

    vutils.save_image(x_ABA.data, x_ABA_path, normalize=True)
    print("[*] Samples saved: {}".format(x_ABA_path))


def generate_with_B(inputs, path, G_AB, G_BA, idx=None):
    x_BA = G_BA(inputs)
    x_BAB = G_AB(x_BA)

    input_path = '{}/{}_input_B.png'.format(path, idx)
    x_BA_path = '{}/{}_x_BA.png'.format(path, idx)
    x_BAB_path = '{}/{}_x_BAB.png'.format(path, idx)

    vutils.save_image(inputs, input_path, normalize=True)
    vutils.save_image(x_BA.data, x_BA_path, normalize=True)
    print("[*] Samples saved: {}".format(x_BA_path))

    vutils.save_image(x_BAB.data, x_BAB_path, normalize=True)
    print("[*] Samples saved: {}".format(x_BAB_path))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def _get_variable(inputs):
    return Variable(inputs.cuda())
