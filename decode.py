import math
import argparse
import abc
from dataclasses import dataclass
from textwrap import wrap


# transmission:
# prologue - packet - wait - prologue - packet - wait
#
# where:
#  prologue: 8*DUTY_CYCLE pulse - 4*DUTY_CYCLE space
#  packet:
#    logic 0: 1*DUTY_CYCLE pulse - 1*DUTY_CYCLE space
#    logic 1: 1*DUTY_CYCLE pulse - 3*DUTY_CYCLE space
#  wait: > 20*DUTY_CYCLE space


DUTY_CYCLE = 330


@dataclass
class DecodedSignal:
    name: str
    data: list[str]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-b', '--binary', action='store_true', help='display the decoded message as binary string')
    parser.add_argument('-d', '--doubles', action='store_true', help='display duplicate consecutive packets')
    parser.add_argument('-t', '--timeout-check', action='store_true', help='validate that each file contains only 1 timeout. Can be enabled when the file is expected to contain codes only for 1 keypress')
    parser.add_argument('-f', '--format', choices=['lirc', 'flipper'], default='lirc', help='input file format. If lirc the input files are expected to contain the dump from `mode2` program')
    parser.add_argument('-s', '--signal-name', help='name of the signal to decode. Only meaningful if --format=flipper')
    parser.add_argument('filenames', nargs='*', help='list of files to decode. They should contain signal data according to the --format option')

    return parser.parse_args()


class Decoder(abc.ABC):
    def decode(self, filename: str, options: argparse.Namespace) -> list[DecodedSignal]:
        if not self._validate(filename, options):
            print(f'{filename} invalid')
            return ''

        signals = self._extract_signals(filename)

        return [
            DecodedSignal(
                name=signal_name,
                data=self._decode(signal_data, options),
            )
            for signal_name, signal_data in signals.items()
        ]

    @abc.abstractmethod
    def _extract_signals(self, filename: str) -> dict[str, list[int]]:
        pass

    @abc.abstractmethod
    def _validate(self, filename: str, options: argparse.Namespace) -> bool:
        pass

    def _decode(self, raw_data: list[int], options: argparse.Namespace) -> list[str]:
        binaries = []
        binary = ''

        i = 0
        while i < len(raw_data) - 1:
            first = raw_data[i]

            if self._is_wait(first):
                binaries.append(binary)
                i += 1
                continue

            i += 1
            second = raw_data[i]

            if self._is_prologue(first, second):
                binaries.append(binary)
                binary = ''
            else:
                binary += self._decode_bit(first, second)

        return [b for b in binaries if len(b) != 0]

    def _normalize(self, pulse: int, space: int) -> (int, int):
        normalized_pulse = math.floor(pulse / DUTY_CYCLE)
        normalized_space = math.floor(space / DUTY_CYCLE)

        return normalized_pulse, normalized_space

    def _is_prologue(self, pulse: int, space: int) -> bool:
        normalized_pulse, normalized_space = self._normalize(pulse, space)
        return normalized_pulse >= 3 and normalized_space >= 3

    def _is_wait(self, signal: int) -> bool:
        return signal > 20 * DUTY_CYCLE

    def _decode_bit(self, pulse: int, space: int) -> str:
        normalized_pulse, normalized_space = self._normalize(pulse, space)
        return '0' if abs(normalized_pulse - normalized_space) < 2 else '1'


class LircDecoder(Decoder):
    def _validate(self, filename: str, options: argparse.Namespace) -> bool:
        found_pulse = False
        found_space = False
        found_timeout = False

        with open(filename) as f:
            for line in f:
                if options.timeout_check and found_timeout:
                    print('timeout should be the last line')
                    return False

                signal = line.split()

                if signal[0] == 'pulse':
                    if found_pulse:
                        print('invalid sequence: 2 pulse in a row')
                        return False
                    found_pulse = True

                if signal[0] == 'space':
                    if not found_pulse or found_space:
                        print('invalid sequence: 2 space in a row')
                        return False
                    found_space = True

                if found_pulse and found_space:
                    found_pulse = False
                    found_space = False

                if signal[0] == 'timeout':
                    found_timeout = True
                    if not options.timeout_check:
                        found_pulse = False
                        found_space = False

        return True

    def _extract_signals(self, filename: str) -> dict[str, list[int]]:
        with open(filename) as f:
            return {filename: [int(line.split()[1]) for line in f]}


class FlipperDecoder(Decoder):
    def _validate(self, filename: str, options: argparse.Namespace) -> bool:
        return True

    def _extract_signals(self, filename: str) -> dict[str, list[int]]:
        signals = {}

        with open(filename) as f:
            line = f.readline()
            while line != '':
                while not line.strip().startswith('#'):
                    line = f.readline()

                signal_name = f.readline().split(':')[1].strip()
                signal_type = f.readline().split(':')[1].strip()
                signal_frequency = int(f.readline().split(':')[1].strip())
                signal_duty_cycle = float(f.readline().split(':')[1].strip())
                signal_data = [int(x) for x in f.readline().split(':')[1].strip().split()]

                signals[signal_name] = signal_data

                line = f.readline()

        return signals


def to_hex(binary: str) -> str:
    binary_bytes = wrap(binary, 8)
    hex_bytes = [f'{hex(int(b, 2))[2:]:02}' for b in binary_bytes]
    return ' '.join(hex_bytes)

    return hex_str


def print_decoded(decoded: DecodedSignal, options: argparse.Namespace) -> None:
    packets = decoded.data if options.binary else [to_hex(d) for d in decoded.data]
    last_packet = ''
    print(decoded.name)
    print(f'packet  data')
    for i, packet in enumerate(packets):
        if options.doubles or packet != last_packet:
            print(f'{i+1:02}/{len(packets):02}   {packet}')
            last_packet = packet


if __name__ == '__main__':
    options = parse_arguments()
    decoder = LircDecoder() if options.format == 'lirc' else FlipperDecoder()

    for filename in options.filenames:
        decoded = decoder.decode(filename, options)
        for decoded_signal in decoded:
            print_decoded(decoded_signal, options)
            print()
