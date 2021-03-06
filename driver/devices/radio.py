# Copyright (C) 2021 Weixuan Zhang, Ghifari Pradana
#
# SPDX-License-Identifier: MIT

import json

from controller import Emitter, Receiver


class IDPEmitter(Emitter):
    def __init__(self, name):
        super().__init__(name)


class IDPReceiver(Receiver):
    def __init__(self, name, sampling_rate):
        super().__init__(name)
        self.enable(sampling_rate)


class IDPRadio:
    def __init__(self, sampling_rate):
        self.emitter = IDPEmitter('emitter')
        self.receiver = IDPReceiver('receiver', sampling_rate)

    @staticmethod
    def encode_message(message: dict) -> bytes:
        """Encode the given dictionary into bytes by first dumping it as JSON

        Args:
            message (dict):The message to send

        Returns:
            bytes: The encoded bytes
        """
        return json.dumps(message).encode('utf-8')

    @staticmethod
    def decode_message(message_bytes: bytes) -> dict:
        """Decode the received bytes as a dictionary

        Args:
            message_bytes (bytes): Received bytes

        Returns:
            dict: The decoded dictionary
        """
        return json.loads(message_bytes.decode('utf-8'))

    def send_message(self, message: dict) -> None:
        """Send the given message

        Args:
            message (dict): The message to send
        """
        self.emitter.send(IDPRadio.encode_message(message))

    def get_message(self) -> dict:
        """Get the latest received message

        Returns:
            dict: The decoded received message
        """
        queue_length = self.receiver.getQueueLength()
        if queue_length == 0:
            return {}

        for _ in range(queue_length - 1):  # get the latest encoded message in the queue
            self.receiver.nextPacket()

        message = IDPRadio.decode_message(self.receiver.getData())
        self.receiver.nextPacket()

        return message
