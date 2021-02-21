from periscope.data.creator import DataCreator


def main():
    protein = '5FCR_A'
    protein = protein[0:4].lower() + protein[-1]

    DataCreator(protein, family='trypsin').ccmpred


if __name__ == '__main__':
    main()
