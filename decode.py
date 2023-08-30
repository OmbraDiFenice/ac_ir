import math
import argparse
import abc


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
    def decode(self, filename: str, options: argparse.Namespace) -> str:
        if not self._validate(filename, options):
            print(f'{filename} invalid')
            return ''

        return self._decode(filename, options)

    @abc.abstractmethod
    def _validate(self, filename: str, options: argparse.Namespace) -> bool:
        pass

    @abc.abstractmethod
    def _decode(self, filename: str, options: argparse.Namespace) -> list[str]:
        pass

    def _normalize(self, first: str, second: str) -> (str, str):
        pulse = int(first.split()[1])
        space = int(second.split()[1])
        normalized_pulse = math.floor(pulse / DUTY_CYCLE)
        normalized_space = math.floor(space / DUTY_CYCLE)

        return normalized_pulse, normalized_space

    def _is_prologue(self, first: str, second: str) -> bool:
        pulse, space = self._normalize(first, second)
        return pulse >= 3 and space >= 3

    def _is_wait(self, line: str) -> bool:
        signal = line.split()
        return signal[0] == 'timeout' or int(signal[1]) > 20 * DUTY_CYCLE

    def _decode_bit(self, first: str, second: str) -> str:
        pulse, space = self._normalize(first, second)
        return '0' if abs(pulse - space) < 2 else '1'


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

    def _decode(self, filename: str, options: argparse.Namespace) -> list[str]:
        binaries = []
        binary = ''

        with open(filename) as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                first = lines[i]

                if self._is_wait(first):
                    binaries.append(binary)
                    i += 1
                    continue

                i += 1
                second = lines[i]

                if self._is_prologue(first, second):
                    binaries.append(binary)
                    binary = ''
                else:
                    binary += self._decode_bit(first, second)

        return [b for b in binaries if len(b) != 0]


class FlipperDecoder(Decoder):
    def _validate(self, filename: str, options: argparse.Namespace) -> bool:
        return True

    def _decode(self, filename: str, options: argparse.Namespace) -> list[str]:
        with open(filename) as f:
            line = f.readline()
            while line != '':
                while not line.strip().startswith('#'):
                    line = f.readline()

                signal_name = f.readline().split(':')[1].strip()
                signal_type = f.readline().split(':')[1].strip()
                signal_frequency = int(f.readline().split(':')[1].strip())
                signal_duty_cycle = float(f.readline().split(':')[1].strip())
                signal_data = f.readline().split(':')[1].strip().split()

                if signal_name == options.signal_name:
                    return self._decode_signal(signal_data, options)

                line = f.readline()

    def _decode_signal(self, lines: list[str], options: argparse.Namespace) -> str:
        binaries = []
        binary = ''

        i = 0
        while i < len(lines) - 1:
            first = lines[i]

            if self._is_wait(first):
                binaries.append(binary)
                i += 1
                continue

            i += 1
            second = lines[i]

            if self._is_prologue(first, second):
                binaries.append(binary)
                binary = ''
            else:
                binary += self._decode_bit(first, second)

        return [b for b in binaries if len(b) != 0]

    def _is_wait(self, signal: str) -> bool:
        return int(signal) > 20 * DUTY_CYCLE

    def _normalize(self, first: str, second: str) -> (str, str):
        pulse = int(first)
        space = int(second)
        normalized_pulse = math.floor(pulse / DUTY_CYCLE)
        normalized_space = math.floor(space / DUTY_CYCLE)

        return normalized_pulse, normalized_space


def to_hex(binary: str) -> str:
    hex_str = ''
    for i, hex_digit in enumerate(hex(int(binary, 2))[2:]):
        hex_str += hex_digit
        if (i + 1) % 2 == 0:
            hex_str += ' '

    return hex_str


def print_decoded(decoded: [str], options: argparse.Namespace) -> None:
    packets = decoded if options.binary else [to_hex(d) for d in decoded]
    last_packet = ''
    print(f'packet  data')
    for i, packet in enumerate(packets):
        if options.doubles or packet != last_packet:
            print(f'{i+1:02}/{len(packets):02}   {packet}')
            last_packet = packet


if __name__ == '__main__':
    options = parse_arguments()
    decoder = LircDecoder() if options.format == 'lirc' else FlipperDecoder()

    for filename in options.filenames:
        print(filename)
        decoded = decoder.decode(filename, options)
        if len(decoded) != 0:
            print_decoded(decoded, options)
        print()
