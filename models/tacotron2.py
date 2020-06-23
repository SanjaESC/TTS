import copy
import torch
from math import sqrt
from torch import nn
from TTS.layers.tacotron2 import Encoder, Decoder, Postnet
from TTS.utils.generic_utils import sequence_mask
from TTS.layers.gst_layers import GST


# TODO: match function arguments with tacotron
class Tacotron2(nn.Module):
    def __init__(self,
                num_chars,
                num_speakers,
                r,
                postnet_output_dim=80,
                decoder_output_dim=80,
                attn_type='original',
                attn_win=False,
                gst=False,
                gst_embedding_dim=512,
                gst_num_heads=4,
                gst_style_tokens=10,
                attn_norm="softmax",
                prenet_type="original",
                prenet_dropout=True,
                forward_attn=False,
                trans_agent=False,
                forward_attn_mask=False,
                location_attn=True,
                attn_K=5,
                separate_stopnet=True,
                bidirectional_decoder=False):
        super(Tacotron2, self).__init__()

        # init parameter
        self.postnet_output_dim = postnet_output_dim
        self.decoder_output_dim = decoder_output_dim
        self.r = r
        self.gst = gst
        self.num_speakers = num_speakers
        self.bidirectional_decoder = bidirectional_decoder
        speaker_embedding_dim = 512 if num_speakers > 1 else 0
        gst_embedding_dim = gst_embedding_dim if self.gst else 0
        decoder_dim = 512+speaker_embedding_dim+gst_embedding_dim
        encoder_dim = 512 if num_speakers > 1 else 512
        proj_speaker_dim = 80 if num_speakers > 1 else 0

        # embedding layer
        self.embedding = nn.Embedding(num_chars, 512, padding_idx=0)
        std = sqrt(2.0 / (num_chars + 512))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        # speaker embedding layer
        if num_speakers > 1:
            self.speaker_embedding = nn.Embedding(num_speakers, speaker_embedding_dim)
            self.speaker_embedding.weight.data.normal_(0, 0.3)

        self.encoder = Encoder(encoder_dim)
        self.decoder = Decoder(decoder_dim, self.decoder_output_dim, r, attn_type, attn_win,
                               attn_norm, prenet_type, prenet_dropout,
                               forward_attn, trans_agent, forward_attn_mask,
                               location_attn, attn_K, separate_stopnet, proj_speaker_dim)

        if self.bidirectional_decoder:
            self.decoder_backward = copy.deepcopy(self.decoder)
        self.postnet = Postnet(self.postnet_output_dim)

        # global style token layers
        if self.gst:
            self.gst_layer = GST(num_mel=80,
                                 num_heads=gst_num_heads,
                                 num_style_tokens=gst_style_tokens,
                                 embedding_dim=gst_embedding_dim)


    @staticmethod
    def shape_outputs(mel_outputs, mel_outputs_postnet, alignments):
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)
        return mel_outputs, mel_outputs_postnet, alignments


    def compute_gst(self, inputs, style_input):
        if isinstance(style_input, dict):
            device = inputs.device
            query = torch.zeros(1, 1, 256).to(device)
            _GST = torch.tanh(self.gst_layer.style_token_layer.style_tokens)

            gst_outputs = torch.zeros(1, 1, 512).to(device)
            for k_token, v_amplifier in style_input.items():
                key = _GST[int(k_token)].unsqueeze(0).expand(1, -1, -1)
                gst_outputs_att = self.gst_layer.style_token_layer.attention(query, key)
                gst_outputs = gst_outputs + gst_outputs_att * v_amplifier

        else:
            gst_outputs = self.gst_layer(style_input)
        embedded_gst = gst_outputs.repeat(1, inputs.size(1), 1)
        return inputs, embedded_gst


    def forward(self, text, text_lengths, mel_specs=None, speaker_ids=None):
        # compute mask for padding
        mask = sequence_mask(text_lengths).to(text.device)
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        if self.num_speakers > 1:
            embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
            embedded_speakers = embedded_speakers.repeat(1, encoder_outputs.size(1), 1)
            if hasattr(self, 'gst'):
                # B x gst_dim
                encoder_outputs, embedded_gst = self.compute_gst(encoder_outputs, mel_specs)
                encoder_outputs = torch.cat([encoder_outputs, embedded_gst, embedded_speakers], dim=-1)
            else:
                encoder_outputs = torch.cat([encoder_outputs, embedded_speakers], dim=-1)
        else:
            if hasattr(self, 'gst'):
                # B x gst_dim
                encoder_outputs, embedded_gst = self.compute_gst(encoder_outputs, mel_specs)
                encoder_outputs = torch.cat([encoder_outputs, embedded_gst], dim=-1)


        decoder_outputs, alignments, stop_tokens = self.decoder(
            encoder_outputs, mel_specs, mask)
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = decoder_outputs + postnet_outputs
        decoder_outputs, postnet_outputs, alignments = self.shape_outputs(
            decoder_outputs, postnet_outputs, alignments)
        if self.bidirectional_decoder:
            decoder_outputs_backward, alignments_backward = self._backward_inference(mel_specs, encoder_outputs, mask)
            return decoder_outputs, postnet_outputs, alignments, stop_tokens, decoder_outputs_backward, alignments_backward
        return decoder_outputs, postnet_outputs, alignments, stop_tokens


    @torch.no_grad()
    def inference(self, text, speaker_ids=None, input_style=None):
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        if self.num_speakers > 1:
            embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
            embedded_speakers = embedded_speakers.repeat(1, encoder_outputs.size(1), 1)
            if hasattr(self, 'gst') and input_style is not None:
                # B x gst_dim
                encoder_outputs, embedded_gst = self.compute_gst(encoder_outputs, input_style)
                encoder_outputs = torch.cat([encoder_outputs, embedded_gst, embedded_speakers], dim=-1)
            else:
                encoder_outputs = torch.cat([encoder_outputs, embedded_speakers], dim=-1)

        else:
            if hasattr(self, 'gst') and input_style is not None:
                # B x gst_dim
                encoder_outputs, embedded_gst = self.compute_gst(encoder_outputs, input_style)
                encoder_outputs = torch.cat([encoder_outputs, embedded_gst], dim=-1)

        decoder_outputs, alignments, stop_tokens = self.decoder.inference(
            encoder_outputs)
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = decoder_outputs + postnet_outputs
        decoder_outputs, postnet_outputs, alignments = self.shape_outputs(
            decoder_outputs, postnet_outputs, alignments)
        return decoder_outputs, postnet_outputs, alignments, stop_tokens


    def inference_truncated(self, text, speaker_ids=None, input_style=None):
        """
        Preserve model states for continuous inference
        """
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference_truncated(embedded_inputs)
        
        if self.num_speakers > 1:
            embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
            embedded_speakers = embedded_speakers.repeat(1, encoder_outputs.size(1), 1)
            if hasattr(self, 'gst') and input_style is not None:
                # B x gst_dim
                encoder_outputs, embedded_gst = self.compute_gst(encoder_outputs, input_style)
                encoder_outputs = torch.cat([encoder_outputs, embedded_gst, embedded_speakers], dim=-1)
            else:
                encoder_outputs = torch.cat([encoder_outputs, embedded_speakers], dim=-1)

        else:
            if hasattr(self, 'gst') and input_style is not None:
                # B x gst_dim
                encoder_outputs, embedded_gst = self.compute_gst(encoder_outputs, input_style)
                encoder_outputs = torch.cat([encoder_outputs, embedded_gst], dim=-1)

        mel_outputs, alignments, stop_tokens = self.decoder.inference_truncated(
            encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
        return mel_outputs, mel_outputs_postnet, alignments, stop_tokens


    def _backward_inference(self, mel_specs, encoder_outputs, mask):
        decoder_outputs_b, alignments_b, _ = self.decoder_backward(
            encoder_outputs, torch.flip(mel_specs, dims=(1,)), mask)
        decoder_outputs_b = decoder_outputs_b.transpose(1, 2)
        return decoder_outputs_b, alignments_b