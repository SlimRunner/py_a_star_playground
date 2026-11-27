import typer

app = typer.Typer()


@app.command()
def maze_001(jump_policy: int, anim: bool = False):
    from weird_maze_2.solve import solve_maze

    solve_maze(jump_policy, anim)


@app.command()
def sokoban(anim: bool = False):
    from sokoban.solve import solve_sokoban

    solve_sokoban(anim)


if __name__ == "__main__":
    app()
