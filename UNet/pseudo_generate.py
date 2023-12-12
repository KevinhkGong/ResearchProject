import subprocess


def run_command(template_command, start_index, end_index):
    for i in range(start_index, end_index + 1):
        # Format the command with the current index
        formatted_command = template_command.format(i,i)

        try:
            result = subprocess.run(formatted_command, check=True, shell=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            print(f"Execution {i}: Success")
            print(result.stdout.decode())
        except subprocess.CalledProcessError as e:
            print(f"Execution {i}: Error")
            print(e.stderr.decode())


# Base command template
command_template = "python predict.py -i data/imgs_test/IDRiD_{:02}.jpg -o optic_disc_pseudolabel/IDRiD_{:02}_pseudo.jpg -m Experiment_Data/IDRiD_Exp/Optic_Disc/checkpoint_epoch231.pth"

# Start and end indices
start_index = 55  # Starting from IDRiD_01.jpg
end_index = 81  # Ending at IDRiD_16.jpg

run_command(command_template, start_index, end_index)
