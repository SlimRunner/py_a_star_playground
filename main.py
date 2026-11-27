import typer

app = typer.Typer()


@app.command()
def maze_001(jump_policy: int, anim: bool = False):
    from weird_maze_2.solve import solve_maze

    solve_maze(jump_policy, anim)


if __name__ == "__main__":
    app()
