# coding: utf-8
import torch
import copy
from torch import nn
from TTS.layers.tacotron import Encoder, Decoder, PostCBHG
from TTS.utils.generic_utils import sequence_mask
from TTS.layers.gst_layers import GST


class Tacotron(nn.Module):
    def __init__(self,
                num_chars,
                num_speakers,
                r=5,
                postnet_output_dim=1025,
                decoder_output_dim=80,
                memory_size=5,
                attn_type='original',
                attn_win=False,
                gst=False,
                gst_embedding_dim=256,
                gst_num_heads=4,
                gst_style_tokens=10,
                attn_norm="sigmoid",
                prenet_type="original",
                prenet_dropout=True,
                forward_attn=False,
                trans_agent=False,
                forward_attn_mask=False,
                location_attn=True,
                attn_K=5,
                separate_stopnet=True,
                bidirectional_decoder=False):
        super(Tacotron, self).__init__()
        self.r = r
        self.decoder_output_dim = decoder_output_dim
        self.postnet_output_dim = postnet_output_dim
        self.gst = gst
        self.num_speakers = num_speakers
        self.bidirectional_decoder = bidirectional_decoder
        speaker_embedding_dim = 128 if num_speakers > 1 else 0
        gst_embedding_dim = gst_embedding_dim if self.gst else 0
        decoder_dim = 512+speaker_embedding_dim+gst_embedding_dim if num_speakers > 1 else 256
        encoder_dim = 512 if num_speakers > 1 else 512
        proj_speaker_dim = 80 if num_speakers > 1 else 0
        # embedding layer
        self.embedding = nn.Embedding(num_chars, 256, padding_idx=0)
        self.embedding.weight.data.normal_(0, 0.3)
        # boilerplate model
        self.encoder = Encoder(encoder_dim)
        self.decoder = Decoder(decoder_dim, decoder_output_dim, r, memory_size, attn_type, attn_win,
                               attn_norm, prenet_type, prenet_dropout,
                               forward_attn, trans_agent, forward_attn_mask,
                               location_attn, attn_K, separate_stopnet,
                               proj_speaker_dim)
        if self.bidirectional_decoder:
            self.decoder_backward = copy.deepcopy(self.decoder)
        self.postnet = PostCBHG(decoder_output_dim)
        self.last_linear = nn.Linear(self.postnet.cbhg.gru_features * 2,
                                     postnet_output_dim)
        # speaker embedding layers
        if num_speakers > 1:
            self.speaker_embedding = nn.Embedding(num_speakers, speaker_embedding_dim)
            self.speaker_embedding.weight.data.normal_(0, 0.3)
            self.speaker_project_mel = nn.Sequential(
                nn.Linear(256, proj_speaker_dim), nn.Tanh())
            self.speaker_embeddings = None
            self.speaker_embeddings_projected = None
        # global style token layers
        if self.gst:
            gst_embedding_dim = 256
            self.gst_layer = GST(num_mel=80,
                                 num_heads=gst_num_heads,
                                 num_style_tokens=gst_style_tokens,
                                 embedding_dim=gst_embedding_dim)

    def _init_states(self):
        self.speaker_embeddings = None
        self.speaker_embeddings_projected = None

    def compute_speaker_embedding(self, speaker_ids):
        if hasattr(self, "speaker_embedding") and speaker_ids is None:
            raise RuntimeError(
                " [!] Model has speaker embedding layer but speaker_id is not provided"
            )
        if hasattr(self, "speaker_embedding") and speaker_ids is not None:
            self.speaker_embeddings = self._compute_speaker_embedding(
                speaker_ids)
            self.speaker_embeddings_projected = self.speaker_project_mel(
                self.speaker_embeddings).squeeze(1)

    def compute_gst(self, inputs, style_input):
        if isinstance(style_input, int):
            query = torch.zeros(1, 1, 128).cuda()
            _GST = torch.tanh(self.gst_layer.style_token_layer.style_tokens)
            key = _GST[style_input].unsqueeze(0).expand(1, -1, -1)
            gst_outputs = self.gst_layer.style_token_layer.attention(query, key)
        else:
            gst_outputs = self.gst_layer(style_input)
        embedded_gst = gst_outputs.repeat(1, inputs.size(1), 1)
        #inputs = self._add_speaker_embedding(inputs, embedded_gst)
        return inputs, embedded_gst

    def forward(self, characters, text_lengths, mel_specs, speaker_ids=None):
        """
        Shapes:
            - characters: B x T_in
            - text_lengths: B
            - mel_specs: B x T_out x D
            - speaker_ids: B x 1
        """
        self._init_states()
        mask = sequence_mask(text_lengths).to(characters.device)
        # B x T_in x embed_dim
        embedded_inputs = self.embedding(characters)
        if self.num_speakers > 1:
            embedded_inputs = embedded_inputs.repeat(1, 1, 2)

        encoder_outputs = self.encoder(embedded_inputs)
        print(embedded_inputs.shape)
        # B x speaker_embed_dim
        if self.num_speakers > 1:
            embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
            embedded_speakers = embedded_speakers.repeat(1, encoder_outputs.size(1), 1)
            print(embedded_speakers.shape)
            if hasattr(self, 'gst'):
                # B x gst_dim
                encoder_outputs, embedded_gst = self.compute_gst(encoder_outputs, mel_specs)
                encoder_outputs = torch.cat([encoder_outputs, embedded_gst, embedded_speakers], dim=-1)
                #encoder_outputs = encoder_outputs + embedded_gst
            else:
                encoder_outputs = torch.cat([encoder_outputs, embedded_speakers], dim=-1)

        else:
            if hasattr(self, 'gst'):
                # B x gst_dim
                encoder_outputs, embedded_gst = self.compute_gst(encoder_outputs, mel_specs)
                encoder_outputs = torch.cat([encoder_outputs, embedded_gst], dim=-1)
                #encoder_outputs = encoder_outputs + embedded_gst

        # decoder_outputs: B x decoder_dim x T_out
        # alignments: B x T_in x encoder_dim
        # stop_tokens: B x T_in
        decoder_outputs, alignments, stop_tokens = self.decoder(
            encoder_outputs, mel_specs, mask)
        # B x T_out x decoder_dim
        postnet_outputs = self.postnet(decoder_outputs)
        # B x T_out x posnet_dim
        postnet_outputs = self.last_linear(postnet_outputs)
        # B x T_out x decoder_dim
        decoder_outputs = decoder_outputs.transpose(1, 2).contiguous()
        if self.bidirectional_decoder:
            decoder_outputs_backward, alignments_backward = self._backward_inference(mel_specs, encoder_outputs, mask)
            return decoder_outputs, postnet_outputs, alignments, stop_tokens, decoder_outputs_backward, alignments_backward
        return decoder_outputs, postnet_outputs, alignments, stop_tokens

    @torch.no_grad()
    def inference(self, characters, speaker_ids=None, input_style=None):
        embedded_inputs = self.embedding(characters)
        encoder_outputs = self.encoder(embedded_inputs)
        if self.num_speakers > 1:
            embedded_speakers = self.speaker_embedding(speaker_ids)[:, None]
            embedded_speakers = embedded_speakers.repeat(1, encoder_outputs.size(1), 1)
            if hasattr(self, 'gst') and input_style is not None:
                # B x gst_dim
                encoder_outputs, embedded_gst = self.compute_gst(encoder_outputs, input_style)
                encoder_outputs = torch.cat([encoder_outputs, embedded_gst, embedded_speakers], dim=-1)
                #encoder_outputs = encoder_outputs + embedded_gst
            else:
                encoder_outputs = torch.cat([encoder_outputs, embedded_speakers], dim=-1)

        else:
            if hasattr(self, 'gst') and input_style is not None:
                # B x gst_dim
                encoder_outputs, embedded_gst = self.compute_gst(encoder_outputs, input_style)
                encoder_outputs = torch.cat([encoder_outputs, embedded_gst], dim=-1)
                #encoder_outputs = encoder_outputs + embedded_gst

        decoder_outputs, alignments, stop_tokens = self.decoder.inference(
            encoder_outputs)
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = self.last_linear(postnet_outputs)
        decoder_outputs = decoder_outputs.transpose(1, 2)
        return decoder_outputs, postnet_outputs, alignments, stop_tokens

    def _backward_inference(self, mel_specs, encoder_outputs, mask):
        decoder_outputs_b, alignments_b, _ = self.decoder_backward(
            encoder_outputs, torch.flip(mel_specs, dims=(1,)), mask,
            self.speaker_embeddings_projected)
        decoder_outputs_b = decoder_outputs_b.transpose(1, 2).contiguous()
        return decoder_outputs_b, alignments_b

    def _compute_speaker_embedding(self, speaker_ids):
        speaker_embeddings = self.speaker_embedding(speaker_ids)
        return speaker_embeddings.unsqueeze_(1)

    @staticmethod
    def _add_speaker_embedding(outputs, speaker_embeddings):
        speaker_embeddings_ = speaker_embeddings.expand(
            outputs.size(0), outputs.size(1), -1)
        outputs = outputs + speaker_embeddings_
        return outputs

    @staticmethod
    def _concat_speaker_embedding(outputs, speaker_embeddings):
        speaker_embeddings_ = speaker_embeddings.expand(
            outputs.size(0), outputs.size(1), -1)
        outputs = torch.cat([outputs, speaker_embeddings_], dim=-1)
        return outputs
