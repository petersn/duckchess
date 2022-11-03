import os
import sys
import signal
import json
import time
import glob
import collections
import traceback
import subprocess

GAME_TARGET = 300

if len(sys.argv) != 2 or sys.argv[1] in ["-h", "--help"]:
    print("Usage: python check_elos.py run-prefix/")
    exit(1)

prefix = sys.argv[1]

def generate_games(output_dir, model_a, model_b, game_count):
    print("Generating games between", model_a, "and", model_b)
    proc = subprocess.Popen(
        [
            #"cargo", "run", "--bin", "compete", "--release", "--", "--playouts", "400",
            prefix + "/compete2", "--playouts", "100",
            "--model1-dir", model_a,
            "--model2-dir", model_b,
            "--output-dir", output_dir,
        ],
        stdout=subprocess.PIPE,
        close_fds=True,
        env=dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true"),
    )
    while game_count:
        l = proc.stdout.readline()
        print("\x1b[92m[%i] LINE:\x1b[0m" % game_count, l)
        if not l:
            raise ValueError("Empty read from process")
        if b"Generated a game" in l:
            game_count -= 1
    time.sleep(3)
    print("\x1b[91mDone!\x1b[0m")
    os.kill(proc.pid, signal.SIGTERM)
    time.sleep(2)
    if proc.poll() is None:
        print("\x1b[91mForcefully killing:\x1b[0m", proc)
        proc.kill()
    proc.wait()
    print("\x1b[91mExited\x1b[0m")

def short_name(s):
    x = int(s.split("-")[-2].split("/")[0])
    return "s%i" % x

def regenerate_report(steps, games):
    subprocess.check_call(["python", "ml/get_elos.py"] + glob.glob(prefix + "/eval-games/*.json"))
    with open("web/template.html") as f:
        template = f.read()
    # Make the table.
    table_data = [[games[a, b] for b in steps] for a in steps]
    table = "<table><tr><th></th>"
    for step in steps:
        table += "<th>" + short_name(step) + "</th>"
    table += "</tr>"
    for i in range(len(steps)):
        table += "<tr><th>" + short_name(steps[i]) + "</th>"
        for j in range(len(steps)):
            if j < i:
                table += "<td></td>"
            elif j == i:
                table += "<td>-</td>"
            else:
                table += "<td>" + str(table_data[i][j]) + "</td>"
        table += "</tr>"
    table += "</table>"

    last_model = int(short_name(steps[-1])[1:])
    cmd = "wc -l %s/step-%03i/games/*.json" % (prefix, last_model)
    print("Running:", cmd)
    output = subprocess.check_output(cmd, shell=True)
    game_count = output.decode().split()[0]
    print("Got:", output)


    with open("web/index.html", "w") as f:
        f.write(template % {
            "table": table,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "game_count": game_count,
        })

def process_name(name):
    if name.startswith("./"):
        name = name[2:]
    if name.endswith("/"):
        name = name[:-1]
    return name

def check():
    # Find all of the games we have so far.
    games = collections.Counter()
    for path in glob.glob(prefix + "/eval-games/*.json"):
        for line in open(path):
            game = json.loads(line)
            engines = tuple(sorted((process_name(game["engine_black"]), process_name(game["engine_white"]))))
            games[engines] += 1

    print("Games:", games)

    # Get a list of all the model pairs we have so far.
    steps = [process_name(step) for step in glob.glob(prefix + "/step-*/model-keras/")]
    steps.sort()

    if games:
        regenerate_report(steps, games)

    # Find all pairs within two steps of each other.
    pairs = []
    for i in range(len(steps)):
        for j in range(len(steps)):
            if i < j <= i + 3:
                pairs.append((steps[i], steps[j]))
    # Sort pairs so that distant ones are last.
    pairs.sort(key=lambda x: abs(steps.index(x[0]) - steps.index(x[1])))
    for a, b in pairs:
        val_a = int(a.split("-")[-2].split("/")[0])
        val_b = int(b.split("-")[-2].split("/")[0])
        assert val_a < val_b <= val_a + 3

        print("   \x1b[95m>", a, "vs", b, "=", games[a, b], "games\x1b[0m")
        if games[a, b] < GAME_TARGET:
            print("Not enough games for", a, "vs", b)
            new_games = max(250, GAME_TARGET - games[a, b])
            generate_games(prefix + "/eval-games/", a, b, new_games)
            try:
                regenerate_report(steps, games)
            except:
                traceback.print_exc()
            break

while True:
    check()
    time.sleep(30)
