"""Run one-variable hyperparameter experiments sequentially, without editing Python."""

import argparse
from pathlib import Path
import subprocess
import sys

from experiments import refresh_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameter', choices=['latent-dim', 'lr', 'beta', 'batch-size', 'image-size', 'depth', 'base-channels'],
                        default='latent-dim')
    parser.add_argument('--values', nargs='+', required=True)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--prefix', default='sweep')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--evaluate', action='store_true', help='Opt-in test evaluation after each case')
    parser.add_argument('--resume-existing', action='store_true')
    args, training_args = parser.parse_known_args()
    if any(arg.split('=')[0] in ('--out', '--resume', '--inspect', '--' + args.parameter)
           for arg in training_args):
        parser.error('Do not supply --out, --resume, --inspect, or the swept parameter as shared arguments')
    if not args.prefix or Path(args.prefix).name != args.prefix or args.prefix in ('.', '..'):
        parser.error('prefix must be a single directory-name component')
    commands = []
    for value in args.values:
        try:
            number = float(value)
            if args.parameter in ('latent-dim', 'batch-size', 'image-size', 'depth', 'base-channels') and not number.is_integer():
                raise ValueError()
            if not 0 < number < float('inf'):
                raise ValueError()
        except ValueError:
            parser.error(f'Invalid positive value: {value}')
        normalized = str(int(number)) if args.parameter in ('latent-dim', 'batch-size', 'image-size', 'depth', 'base-channels') else str(number)
        output = args.runs / f'{args.prefix}_{args.parameter}_{normalized}'
        command = [sys.executable, 'train.py', *training_args, '--out', str(output)]
        if args.resume_existing and (output / 'last.pt').is_file():
            command += ['--resume', str(output / 'last.pt')]
        command += ['--' + args.parameter, normalized]
        commands.append((command, output))
    if len({str(output) for _, output in commands}) != len(commands):
        parser.error('Duplicate parameter values would reuse an output directory')
    for command, output in commands:
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
            if args.evaluate:
                subprocess.run([sys.executable, 'evaluate.py', '--checkpoint', str(output / 'best.pt')], check=True)
    if not args.dry_run:
        refresh_catalog(args.runs, plots=True)


if __name__ == '__main__':
    main()
