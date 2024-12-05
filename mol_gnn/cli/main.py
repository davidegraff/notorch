import click

import mol_gnn.cli.train as train_cli

@click.command
def main():
    print("hello from the C L I!")


main = click.Group("main", train_cli.train)

if __name__ == "__main__":
    main()
