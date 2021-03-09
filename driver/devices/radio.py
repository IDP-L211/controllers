# Copyright (C) 2021 Weixuan Zhang, Ghifari Pradana
#
# SPDX-License-Identifier: MIT

import json
from typing import Union

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
        self.received_cache = {}
        self.message_draft = {}

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
        """Add more message to be sent

        Args:
            message (dict): The key-value pair to be appended to the message
        """
        self.message_draft.update(message)

    def dispatch_message(self) -> None:
        """Send the given message

        This should only be called once per timestep
        """
        self.emitter.send(IDPRadio.encode_message(self.message_draft))
        self.message_draft = {}

    def get_message(self) -> dict:
        """Get the latest received message

        This can be called multiple times per timestep

        Returns:
            dict: The decoded received message
        """
        queue_length = self.receiver.getQueueLength()
        if queue_length < 2:
            # only True at the start of the simulation or get_message been called more than once in a single timestep
            return self.received_cache

        for _ in range(queue_length - 1):  # remove all but the latest message from the queue
            self.receiver.nextPacket()

        message = IDPRadio.decode_message(self.receiver.getData())
        # this message will remain in queue, until the next timestep, where a new message will be inserted to queue
        self.received_cache = message

        return message

    def get_other_bot_position(self) -> Union[None, list]:
        return self.get_message().get('position')

    def get_other_bot_bearing(self) -> Union[None, list]:
        return self.get_message().get('bearing')

    def get_other_bot_vertices(self) -> Union[None, list]:
        return self.get_message().get('vertices')

    def get_other_bot_collected(self) -> Union[None, list]:
        return self.get_message().get('collected')

    def get_other_bot_target_pos(self) -> Union[None, list]:
        return self.get_message().get('target')
