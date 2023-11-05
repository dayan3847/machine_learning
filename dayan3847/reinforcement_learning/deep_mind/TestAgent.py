from Agent import KnowledgeModel

if __name__ == '__main__':
    f_shape = (240, 320, 3)
    # f = np.random.rand(*(2, 4, 6))
    k = KnowledgeModel(
        11,
        f_shape,
        4,
    )
