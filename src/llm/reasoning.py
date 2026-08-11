# """
# LLM Semantic Labeling Module
# =============================
#
# Stage 5 of the training pipeline:
#   - Receives cluster summaries from the Cluster Summarization Module
#   - Prompts GPT-4 as a cybersecurity expert
#   - Produces structured output: tactic, tactic_id, confidence (P_LLM), reasoning
#   - Populates the Knowledge Base
#
# Also used during inference for selective validation and during
# streaming for re-labeling when drift or new clusters are detected.
# """
#
# import json
# import os
# import logging
#
# import numpy as np
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# class LLMLabeler:
#     """LLM-based cluster labeler using GPT-4.
#
#     Produces structured tactic assignments for the Knowledge Base.
#     """
#
#     def __init__(self, api_key=None, model='gpt-4'):
#         self.api_key = api_key or os.getenv('OPENAI_API_KEY')
#         self.model = model
#         self.client = None
#
#         try:
#             from openai import OpenAI
#             self.client = OpenAI(api_key=self.api_key)
#             logger.info(f"LLMLabeler initialized with model={model}")
#         except ImportError:
#             logger.warning("OpenAI library not installed. Using mock labeler.")
#         except Exception as e:
#             logger.warning(f"OpenAI init failed: {e}. Using mock labeler.")
#
#     def label_cluster(self, cluster_summary, cluster_weight):
#         """Label a single cluster using LLM reasoning.
#
#         Args:
#             cluster_summary: dict with 'text' and 'stats' from ClusterSummarizer
#             cluster_weight: float, DP-GMM weight for this cluster (P_GMM prior)
#
#         Returns:
#             dict with: tactic, tactic_id, p_llm, reasoning
#         """
#         summary_text = cluster_summary['text']
#         prompt = self._build_labeling_prompt(summary_text, cluster_weight)
#
#         if self.client is None:
#             # return self._mock_label(cluster_summary)
#             print(cluster_summary)
#
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system",
#                      "content": "You are a cybersecurity expert specializing in "
#                                 "MITRE ATT&CK framework analysis of network traffic."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.3,
#                 max_tokens=400,
#             )
#
#             response_text = response.choices[0].message.content
#             result = self._parse_response(response_text)
#             return result
#
#         except Exception as e:
#             logger.error(f"LLM API error: {e}")
#             # return self._mock_label(cluster_summary)
#
#     def _build_labeling_prompt(self, summary_text, cluster_weight):
#         """Build the structured prompt for cluster labeling.
#
#         Gives GPT-4 specific network-level indicators for each tactic
#         so it can match cluster statistics to attack patterns.
#         """
#
#         return f"""You are a cybersecurity expert analyzing network flow data.
#         Provide TWO levels of analysis:
#
#         1. BEHAVIOUR: Describe the observed network behaviour in 1-2 sentences.
#         2. TACTICS: Map to MITRE ATT&CK tactics from the list below.
#
#         === CRITICAL DECISION FEATURES (ranked by discriminability) ===
#
#         1. TCP_FLAGS (MOST important):
#            - Value ~2 (SYN only) = SYN flood attack → Impact
#            - Value ~16-22 (SYN+ACK+PSH etc.) = normal/full connections → NOT Impact
#
#         2. OUT_BYTES:
#            - Near zero (0-10 bytes) = no data sent, likely SYN flood → Impact
#            - Very high (>100K bytes) = data transfer → Exfiltration
#            - Low (40-100 bytes) = probing/scanning → Reconnaissance
#
#         3. Direction ratio (src_bps / dst_bps):
#            - Very high (>500) = extreme asymmetry, SYN floods → Impact
#            - Moderate (100-400) = some asymmetry → Recon or Exfil
#            - Low (<100) = balanced → Benign
#
#         4. Fan-out (unique destinations per source):
#            - Very high (>40) = many targets → Exfiltration or Benign
#            - Low (<15) = focused → Impact or Reconnaissance
#
#         5. Destination port:
#            - Low (<200) = well-known services → Impact (targeting services)
#            - High (>10000) = random/high ports → Benign or Reconnaissance
#            - Specific (2000-5000) = application-specific → Exfiltration
#
#         6. MIN_TTL:
#            - High (>55) = close/direct source → Impact
#            - Low (<20) = distant/proxied → Benign
#
#         === FEW-SHOT EXAMPLES (from real data) ===
#
#         EXAMPLE 1 — Impact (SYN flood DDoS/DoS):
#           TCP flags: 2.0 (SYN only) ← KEY INDICATOR
#           Out bytes: 0 (no data payload)
#           Direction ratio: 976 (extreme asymmetry)
#           Dst port: 98 (well-known services)
#           MIN_TTL: 61 | PPS: 350 | Fan-out: 9.5
#           → Impact — SYN-only flags + zero outbound data + extreme direction ratio
#             = classic SYN flood denial-of-service
#
#         EXAMPLE 2 — Reconnaissance (network scanning):
#           TCP flags: 21.3 (SYN+ACK+PSH — full TCP connections)
#           Out bytes: 84 (tiny payload)
#           PPS: 15,122 (high rate)
#           Dst port: 14,959 (mixed high ports)
#           Fan-in: 7.5 (highest) | Fan-out: 9.5
#           → Reconnaissance — full TCP handshakes + tiny payload + high PPS
#             = systematic service/port scanning
#
#         EXAMPLE 3 — Exfiltration (data theft):
#           TCP flags: 16.0 (normal TCP)
#           Out bytes: 728,310 (massive outbound data)
#           PPS: 45,623 (very high throughput)
#           Bytes/pkt: 274 (large packets)
#           Fan-out: 49.2 (very high — many destinations)
#           Dst port: 2,525 (specific application port)
#           → Exfiltration — massive outbound bytes + large packets + many destinations
#             = active data exfiltration
#
#         EXAMPLE 4 — None (benign traffic):
#           TCP flags: 18.7 (normal TCP)
#           Out bytes: 54,507 (moderate)
#           MIN_TTL: 16 (lowest — distant sources)
#           Dst port: 23,438 (high/random ports)
#           Fan-out: 39.1 (high — normal browsing)
#           PPS: 8,630 (moderate)
#           → None — moderate values + low TTL + high ports = normal network activity
#
#         === NOW ANALYZE THIS CLUSTER ===
#
#         {summary_text}
#
#         Cluster prior weight: {cluster_weight:.4f}
#
#         Allowed tactics (network-observable ONLY):
#         - Impact (TA0040): SYN-only TCP flags (~2), near-zero out_bytes, extreme
#           direction ratio (>500), low dst ports (<200), high TTL (>55)
#         - Reconnaissance (TA0043): high TCP flags (~21), tiny out_bytes (<100),
#           high PPS, mixed ports, high fan-in
#         - Exfiltration (TA0010): normal TCP flags (~16), massive out_bytes (>100K),
#           large packets, high fan-out (>40)
#         - Command and Control (TA0011): normal TCP flags, long duration, one-to-one,
#           balanced direction, low PPS
#         - Discovery (TA0007): ICMP present, moderate probing
#         - Lateral Movement (TA0008): distributed internal connectivity
#         - None: normal TCP flags (~19), moderate values, low TTL (<20), high dst
#           ports (>10K)
#
#         MOST IMPORTANT RULE: Check TCP_FLAGS value first.
#           TCP_FLAGS ~2 (SYN only) → almost certainly Impact (SYN flood)
#           TCP_FLAGS ~16-22 (normal TCP) → then check OUT_BYTES to distinguish others
#
#         Respond ONLY in JSON:
#         {{
#             "behaviour_label": "<3-5 words>",
#             "behaviour_description": "<1-2 sentences>",
#             "tactics": [
#                 {{
#                     "tactic": "<from list above>",
#                     "tactic_id": "<TA####>",
#                     "technique_id": "<T####>",
#                     "technique_name": "<name>",
#                     "confidence": <0.0-1.0>
#                 }}
#             ],
#             "reasoning": "<cite TCP_FLAGS, OUT_BYTES, direction_ratio values>"
#         }}"""
#         # return f"""You are a cybersecurity expert. Analyze this network traffic cluster
#         # and provide TWO levels of analysis:
#         #
#         # 1. BEHAVIOUR: Describe the observed network behaviour pattern in 1-2 sentences.
#         #    Focus on what the traffic IS DOING (e.g., "systematic port probing",
#         #    "high-volume flooding", "steady bidirectional communication"), not what
#         #    attack it might be.
#         #
#         # 2. TACTICS: Map the described behaviour to one or more MITRE ATT&CK tactics.
#         #    A single behaviour CAN map to multiple tactics. Assign confidence to each.
#         #
#         # {summary_text}
#         #
#         # Cluster prior weight (from DP-GMM): {cluster_weight:.4f}
#         #
#         # Use these network-level indicators for tactic mapping:
#         #
#         # - Reconnaissance (TA0043): many destinations from few sources, short flows,
#         #   small packets, well-known dst ports, high fan-out
#         # - Initial Access (TA0001): targeting well-known ports, moderate-large payloads,
#         #   connection attempts, high packet count to few targets
#         # - Execution (TA0002): large payload packets, unusual TCP flags, short-medium duration
#         # - Persistence (TA0003): long-duration, one-to-one, periodic, low packet rate
#         # - Discovery (TA0007): ICMP prevalent, moderate fan-out, multiple protocols
#         # - Lateral Movement (TA0008): distributed connectivity, ephemeral ports, spreading
#         # - Command and Control (TA0011): long duration, few unique IPs, balanced traffic,
#         #   steady rate, DNS/HTTP protocol
#         # - Exfiltration (TA0010): outbound-heavy, large bytes, large packets
#         # - Impact (TA0040): very high packet/throughput, short duration, SYN-heavy flags
#         # - None: large proportion (>20%), close to global averages, normal patterns
#         #
#         # IMPORTANT: Cite specific numeric values from the summary in your reasoning.
#         #
#         # Respond ONLY in JSON format:
#         # {{
#         #     "behaviour_label": "<short label, 3-5 words, e.g. 'high-volume flooding'>",
#         #     "behaviour_description": "<1-2 sentence description of what the traffic is doing>",
#         #     "tactics": [
#         #         {{
#         #             "tactic": "<tactic name>",
#         #             "tactic_id": "<TA####>",
#         #             "technique_id": "<T####>",
#         #             "technique_name": "<technique name>",
#         #             "confidence": <0.0-1.0>
#         #         }}
#         #     ],
#         #     "reasoning": "<2-3 sentences citing specific numbers from the summary>"
#         # }}"""
#
#         # return f"""
#         # You are a cybersecurity expert specializing in network traffic analysis.
#         #
#         # Analyze the following network traffic cluster summary and estimate how strongly this cluster matches each relevant MITRE ATT&CK tactic.
#         #
#         # {summary_text}
#         #
#         # Cluster prior weight from DP-GMM: {cluster_weight:.4f}
#         #
#         # A cluster may represent more than one tactic. Do NOT force it into only one tactic.
#         # Instead, provide a probability distribution over the most relevant tactics. Each probability must be written with exactly 3 decimal places.
#         #
#         # Use the following network-level indicators as guidance:
#         #
#         # - Reconnaissance (TA0043): many unique destinations from few sources, port scanning, very short flows, small packets, high fan-out, low bytes per flow, common destination ports.
#         #
#         # - Initial Access (TA0001): targeting exposed services such as 80, 443, 22, or 445, connection attempts, moderate duration, exploit-like or fuzzing traffic, repeated attempts to few destinations.
#         #
#         # - Execution (TA0002): payload delivery patterns, moderate or large packets, unusual TCP flags, short-to-medium duration, one-to-one or one-to-few communication.
#         #
#         # - Persistence (TA0003): long-duration flows, periodic or repeated traffic, low packet rate, one-to-one communication, balanced or inbound-heavy traffic.
#         #
#         # - Discovery (TA0007): probing behavior, ICMP or multiple protocols, moderate fan-out, short-to-medium flows, internal service discovery patterns.
#         #
#         # - Lateral Movement (TA0008): internal source and destination communication, multiple internal pairs, TCP-dominant traffic, spreading behavior, use of service ports or ephemeral ports.
#         #
#         # - Command and Control (TA0011): long or repeated connections, few unique IPs, balanced bidirectional traffic, steady byte rate, DNS/HTTP-like communication, low flag variability.
#         #
#         # - Exfiltration (TA0010): outbound-heavy traffic, large source-to-destination bytes, large packets, moderate-to-long duration, high upload ratio.
#         #
#         # - Impact (TA0040): DoS/DDoS patterns, very high packet count, high throughput, many-to-one or one-to-many traffic, SYN-heavy behavior, short intense flows.
#         #
#         # - Benign: large cluster weight, values close to global averages, balanced traffic, normal ports, typical TTL values, no extreme behavior.
#         #
#         # Important rules:
#         # 1. Use the actual numeric values in the summary, such as bytes, packets, duration, ports, unique sources/destinations, and ratios.
#         # 2. Assign probabilities to multiple tactics when the evidence overlaps.
#         # 3. Probabilities must sum to 1.0.
#         # 4. Include only tactics with meaningful evidence. Do not include every tactic unless needed.
#         # 5. If the cluster is unclear, distribute probability across possible tactics and explain the uncertainty.
#         # 6. Treat the DP-GMM cluster prior weight as supporting evidence only, not the main deciding factor.
#         #
#         # Respond ONLY in JSON format:
#         #
#         # {{
#         #   "primary_tactic": "<most likely tactic name>",
#         #   "primary_tactic_id": "<TA#### or Benign>",
#         #   "tactic_probabilities": [
#         #     {{
#         #       "tactic": "<tactic name>",
#         #       "tactic_id": "<TA#### or Benign>",
#         #       "probability": <0.0-1.0>,
#         #       "evidence": "<short evidence using specific numeric values from the summary>"
#         #     }}
#         #   ],
#         #   "confidence": <0.0-1.0>,
#         #   "reasoning": "<2-4 sentences explaining why the probabilities were assigned, citing specific numeric values from the summary>"
#         # }}
#         # """
#
#         #
#         # return f"""You are a cybersecurity expert specializing in network traffic analysis.
#         # Analyze the following network traffic cluster summary and assign the most
#         # likely MITRE ATT&CK tactic based on the concrete network-level indicators.
#         #
#         # {summary_text}
#         #
#         # Cluster prior weight (from DP-GMM): {cluster_weight:.4f}
#         #
#         # Use these network-level indicators to determine the tactic:
#         #
#         # - Reconnaissance (TA0043): Many unique destinations from few sources (port scanning),
#         #   very short flows (<10ms), small packets, high fan-out, destination ports targeting
#         #   well-known range (0-1023), low bytes per flow
#         #
#         # - Initial Access (TA0001): Targeting well-known dst ports (80, 443, 22, 445),
#         #   exploit payloads = moderate-large packets, TCP flags showing connection attempts,
#         #   moderate duration, fuzzing = high packet count to few destinations
#         #
#         # - Execution (TA0002): Large payload packets (shellcode delivery), unusual TCP flags,
#         #   short-medium duration, one-to-one or one-to-few connectivity
#         #
#         # - Persistence (TA0003): Long-duration flows, one-to-one connectivity, periodic traffic
#         #   patterns, low packet rate, backdoor = inbound-heavy or balanced traffic
#         #
#         # - Discovery (TA0007): ICMP prevalent, moderate fan-out, probing multiple protocols,
#         #   balanced traffic, short-medium duration
#         #
#         # - Lateral Movement (TA0008): Internal IP ranges, multiple source-destination pairs,
#         #   ephemeral source ports, TCP-dominant, spreading pattern
#         #
#         # - Command and Control (TA0011): Long duration, one-to-one connectivity, very few
#         #   unique IPs (1-3 each side), balanced bidirectional traffic, low TCP flag variability,
#         #   moderate steady byte rate, DNS or HTTP app protocol
#         #
#         # - Exfiltration (TA0010): Outbound-heavy traffic (high src-to-dst throughput ratio),
#         #   large outbound bytes, moderate-long duration, large packets
#         #
#         # - Impact (TA0040): DoS/DDoS = many-to-one or one-to-many, very high packet counts,
#         #   high throughput, short duration, SYN-heavy TCP flags, very high fan-in
#         #
#         # - Benign: Large proportion of total traffic (>20%), values close to global averages,
#         #   balanced traffic, normal port distribution, typical TTL values, no extreme features
#         #
#         # IMPORTANT: Use the actual numeric values provided (bytes, packets, milliseconds)
#         # to make your assessment. A cluster with 50,000 bytes average is very different
#         # from one with 50 bytes. Duration of 5ms is scanning; duration of 30,000ms
#         # is a persistent connection.
#         #
#         # Respond ONLY in JSON format:
#         # {{
#         #     "tactic": "<tactic name>",
#         #     "tactic_id": "<TA####>",
#         #     "technique_id": "<T####>",
#         #     "technique_name": "<technique name>",
#         #     "confidence": <0.0-1.0>,
#         #     "reasoning": "<2-3 sentences citing specific numeric values from the summary>"
#         # }}"""
#
# #         return f"""You are a cybersecurity expert specializing in network traffic analysis.
# # Analyze the following network traffic cluster summary and assign the most
# # likely MITRE ATT&CK tactic based on the concrete network-level indicators.
# #
# # {summary_text}
# #
# # Cluster prior weight (from DP-GMM): {cluster_weight:.4f}
# #
# # Use these network-level indicators to determine the tactic:
# #
# # - Reconnaissance (TA0043): Many unique destinations from few sources (port scanning),
# #   very short flows (<10ms), small packets, high fan-out, destination ports targeting
# #   well-known range (0-1023), low bytes per flow
# #
# # - Initial Access (TA0001): Targeting well-known dst ports (80, 443, 22, 445),
# #   exploit payloads = moderate-large packets, TCP flags showing connection attempts,
# #   moderate duration, fuzzing = high packet count to few destinations
# #
# # - Execution (TA0002): Large payload packets (shellcode delivery), unusual TCP flags,
# #   short-medium duration, one-to-one or one-to-few connectivity
# #
# # - Persistence (TA0003): Long-duration flows, one-to-one connectivity, periodic traffic
# #   patterns, low packet rate, backdoor = inbound-heavy or balanced traffic
# #
# # - Discovery (TA0007): ICMP prevalent, moderate fan-out, probing multiple protocols,
# #   balanced traffic, short-medium duration
# #
# # - Lateral Movement (TA0008): Internal IP ranges, multiple source-destination pairs,
# #   ephemeral source ports, TCP-dominant, spreading pattern
# #
# # - Command and Control (TA0011): Long duration, one-to-one connectivity, very few
# #   unique IPs (1-3 each side), balanced bidirectional traffic, low TCP flag variability,
# #   moderate steady byte rate, DNS or HTTP app protocol
# #
# # - Exfiltration (TA0010): Outbound-heavy traffic (high src-to-dst throughput ratio),
# #   large outbound bytes, moderate-long duration, large packets
# #
# # - Impact (TA0040): DoS/DDoS = many-to-one or one-to-many, very high packet counts,
# #   high throughput, short duration, SYN-heavy TCP flags, very high fan-in
# #
# # - Benign: Large proportion of total traffic (>20%), values close to global averages,
# #   balanced traffic, normal port distribution, typical TTL values, no extreme features
# #
# # IMPORTANT: Use the actual numeric values provided (bytes, packets, milliseconds)
# # to make your assessment. A cluster with 50,000 bytes average is very different
# # from one with 50 bytes. Duration of 5ms is scanning; duration of 30,000ms
# # is a persistent connection.
# #
# # Respond ONLY in JSON format:
# # {{
# #     "tactic": "<tactic name>",
# #     "tactic_id": "<TA####>",
# #     "confidence": <0.0-1.0>,
# #     "reasoning": "<2-3 sentences citing specific numeric values from the summary>"
# # }}"""
#
#
#     #     return f"""You are a cybersecurity expert. Analyze the following network traffic
#     # cluster and assign the most likely MITRE ATT&CK tactic.
#     #
#     # {summary_text}
#     #
#     # Cluster prior weight (from DP-GMM): {cluster_weight:.4f}
#     #
#     # Based on the feature patterns, behavioural signature, and network topology,
#     # assign a MITRE ATT&CK tactic. Consider these tactics:
#     # - Reconnaissance (TA0043): scanning, probing
#     # - Resource Development (TA0042): infrastructure setup
#     # - Initial Access (TA0001): exploitation, phishing
#     # - Execution (TA0002): code execution
#     # - Persistence (TA0003): maintaining access
#     # - Privilege Escalation (TA0004): gaining higher privileges
#     # - Defense Evasion (TA0005): avoiding detection
#     # - Credential Access (TA0006): stealing credentials
#     # - Discovery (TA0007): exploring the environment
#     # - Lateral Movement (TA0008): moving through the network
#     # - Collection (TA0009): gathering data
#     # - Command and Control (TA0011): C2 communication
#     # - Exfiltration (TA0010): stealing data
#     # - Impact (TA0040): disruption, DoS
#     # - Benign: normal traffic
#     #
#     # Respond ONLY in JSON format:
#     # {{
#     #     "tactic": "<tactic name>",
#     #     "tactic_id": "<TA####>",
#     #     "confidence": <0.0-1.0>,
#     #     "reasoning": "<2-3 sentence explanation of why this tactic fits>"
#     # }}"""
#
#     def _parse_response(self, response_text):
#         """Parse JSON response from LLM."""
#         # Strip markdown code fences if present
#         text = response_text.strip()
#         if text.startswith('```'):
#             text = text.split('\n', 1)[1] if '\n' in text else text[3:]
#         if text.endswith('```'):
#             text = text[:-3]
#         text = text.strip()
#         if text.startswith('json'):
#             text = text[4:].strip()
#
#         # try:
#         #     result = json.loads(text)
#         #     return {
#         #         'tactic': result.get('tactic', 'Unknown'),
#         #         'tactic_id': result.get('tactic_id', 'Unknown'),
#         #         # 'technique_id': result.get('technique_id', 'Unknown'),
#         #         # 'technique_name': result.get('technique_name', 'Unknown'),
#         #         'p_llm': float(result.get('confidence', 0.5)),
#         #         'reasoning': result.get('reasoning', ''),
#         #     }
#         # except json.JSONDecodeError:
#         #     logger.warning(f"Could not parse LLM response: {text[:100]}...")
#         #     return {
#         #         'tactic': 'Unknown',
#         #         'tactic_id': 'Unknown',
#         #         # 'technique_id': 'Unknown',
#         #         # 'technique_name': 'Unknown',
#         #         'p_llm': 0.5,
#         #         'reasoning': 'Parse error',
#         #     }
#         try:
#             result = json.loads(text)
#
#             tactics = result.get('tactics', [])
#             if not tactics:
#                 tactics = [{'tactic': 'Unknown', 'tactic_id': 'Unknown',
#                             'technique_id': 'Unknown', 'technique_name': 'Unknown',
#                             'confidence': 0.5}]
#
#             # Sort by confidence, primary = highest
#             tactics = sorted(tactics, key=lambda t: t.get('confidence', 0),
#                              reverse=True)
#             primary = tactics[0]
#
#             return {
#                 'behaviour_label': result.get('behaviour_label', 'Unknown'),
#                 'behaviour_description': result.get('behaviour_description', ''),
#                 'tactics': tactics,
#                 'tactic': primary.get('tactic', 'Unknown'),
#                 'tactic_id': primary.get('tactic_id', 'Unknown'),
#                 'technique_id': primary.get('technique_id', 'Unknown'),
#                 'technique_name': primary.get('technique_name', 'Unknown'),
#                 'p_llm': float(primary.get('confidence', 0.5)),
#                 'reasoning': result.get('reasoning', ''),
#             }
#         except json.JSONDecodeError:
#             logger.warning(f"Could not parse LLM response: {text[:100]}...")
#             return self._fallback_result()
#
#     def _fallback_result(self):
#         return {
#             'behaviour_label': 'Unknown',
#             'behaviour_description': '',
#             'tactics': [{'tactic': 'Unknown', 'tactic_id': 'Unknown',
#                          'technique_id': 'Unknown', 'technique_name': 'Unknown',
#                          'confidence': 0.5}],
#             'tactic': 'Unknown', 'tactic_id': 'Unknown',
#             'technique_id': 'Unknown', 'technique_name': 'Unknown',
#             'p_llm': 0.5, 'reasoning': 'Parse error',
#         }
#
#     def label_all_clusters(self, cluster_summaries, cluster_weights):
#         """Label all clusters and return structured results.
#
#         Args:
#             cluster_summaries: dict of {k: summary_dict} from ClusterSummarizer
#             cluster_weights: dict of {k: weight} from DP-GMM
#
#         Returns:
#             labels: dict of {k: label_result_dict}
#         """
#         labels = {}
#         for k, summary in cluster_summaries.items():
#             weight = cluster_weights.get(k, 0.0)
#             result = self.label_cluster(summary, weight)
#             labels[k] = result
#             # logger.info(f"Cluster {k}: {result['primary_tactic']} "
#             #              f"(P_LLM={result['p_llm']:.2f})")
#             logger.info(f"Cluster {k}: [{result['behaviour_label']}] -> "
#                         f"{result['tactic']} (P_LLM={result['p_llm']:.2f})")
#         return

"""
LLM Behaviour Labeler
========================

Labels clusters with BEHAVIOUR names from a predefined list.
Does NOT assign tactics — that's done by rule-based mapping downstream.

The LLM picks from a constrained vocabulary (multiple choice, not open-ended):
  syn_flooding, volumetric_flooding, port_scanning, service_probing,
  data_transfer, periodic_communication, normal_browsing, connection_attempts
"""

import json
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Predefined behaviour vocabulary
BEHAVIOUR_LIST = [
    'syn_flooding',
    'volumetric_flooding',
    'port_scanning',
    'service_probing',
    'data_transfer',
    'periodic_communication',
    'normal_browsing',
    'connection_attempts',
]


class LLMLabeler:
    """Labels clusters with behaviour names from predefined list."""

    def __init__(self, api_key=None, model='gpt-4'):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.client = None

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"LLMLabeler initialized with model={model}")
        except (ImportError, Exception) as e:
            logger.warning(f"OpenAI not available: {e}. Using mock labeler.")

    def label_cluster(self, cluster_summary, cluster_weight):
        """Label a single cluster with behaviour(s).

        Returns:
            dict with: behaviours (list), primary_behaviour, confidence, reasoning
        """
        summary_text = cluster_summary.get('text', '')
        prompt = self._build_prompt(summary_text, cluster_weight)

        if self.client is None:
            return self._mock_label(cluster_summary)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a network traffic behaviour analyst. "
                                "You classify traffic patterns into predefined "
                                "behaviour categories. You do NOT assign attack "
                                "labels or MITRE tactics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=400,
            )
            return self._parse_response(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._mock_label(cluster_summary)

    def _build_prompt(self, summary_text, cluster_weight):
        return f"""Analyze this network traffic cluster and assign ONE or MORE
behaviour labels from the list below.

ALLOWED LABELS (pick ONLY from this list):
  syn_flooding, volumetric_flooding, port_scanning, service_probing,
  data_transfer, periodic_communication, normal_browsing, connection_attempts

=== DECISION RULES (follow this order) ===

STEP 1 — Check TCP_FLAGS value first:
  TCP_FLAGS < 4 (SYN only):
    - If direction_ratio > 100 → syn_flooding
    - If few destinations (≤5) and few sources (≤3) → connection_attempts
    - Otherwise → syn_flooding

  TCP_FLAGS ≥ 4 (normal TCP) → go to STEP 2

STEP 2 — Check OUT_BYTES:
  OUT_BYTES > 200,000:
    → data_transfer (definite large data movement)

  OUT_BYTES 10,000-200,000:
    - If MIN_TTL < 30 → normal_browsing (benign users have low TTL ~16)
    - If MIN_TTL ≥ 30 AND fan_out > 30 → data_transfer
    - Otherwise → normal_browsing

  OUT_BYTES 200-10,000:
    - If MIN_TTL < 25 → normal_browsing
    - If MIN_TTL > 50 AND direction_ratio > 100 → volumetric_flooding
    - If MIN_TTL 25-50 AND packets_per_second > 5000 → port_scanning
    - If MIN_TTL 25-50 → service_probing

  OUT_BYTES < 200:
    - If MIN_TTL < 25 → normal_browsing
    - If MIN_TTL > 50 AND direction_ratio > 100 → volumetric_flooding
    - If MIN_TTL 25-50 AND packets_per_second > 5000 → port_scanning
    - If MIN_TTL 25-50 → service_probing

STEP 3 — Safety overrides:
  - Cluster > 5% of total traffic + TCP_FLAGS > 8 → normal_browsing
  - Any data_transfer assignment + MIN_TTL < 25 → override to normal_browsing

=== KEY THRESHOLDS (from dataset analysis) ===
  MIN_TTL:  Benign=16, Recon=27, Exfil=39, Impact=61
  TCP_FLAGS: Impact=2 (SYN), Exfil=16, Benign=19, Recon=21
  OUT_BYTES: Impact=3, Recon=84, Benign=54507, Exfil=728310

=== FEW-SHOT EXAMPLES ===

Cluster A — syn_flooding:
  TCP_FLAGS=2.0, OUT_BYTES=0, MIN_TTL=61, dir_ratio=976 → syn_flooding

Cluster B — port_scanning:
  TCP_FLAGS=21.3, OUT_BYTES=84, MIN_TTL=27, PPS=15122 → port_scanning

Cluster C — data_transfer:
  TCP_FLAGS=16.0, OUT_BYTES=728310, MIN_TTL=39, fan_out=49 → data_transfer

Cluster D — normal_browsing:
  TCP_FLAGS=18.7, OUT_BYTES=54507, MIN_TTL=16, proportion=15% → normal_browsing

=== CLUSTER TO ANALYZE ===

{summary_text}

Cluster weight: {cluster_weight:.4f}

Respond ONLY in JSON:
{{
    "behaviours": [
        {{"label": "<from allowed list>", "confidence": <0.0-1.0>}}
    ],
    "reasoning": "<cite TCP_FLAGS, OUT_BYTES, MIN_TTL values>"
}}"""

    def _parse_response(self, response_text):
        text = response_text.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1] if '\n' in text else text[3:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()
        if text.startswith('json'):
            text = text[4:].strip()

        try:
            result = json.loads(text)
            behaviours = result.get('behaviours', [])

            # Validate labels against allowed list
            valid = []
            for b in behaviours:
                label = b.get('label', '')
                if label in BEHAVIOUR_LIST:
                    valid.append(b)
                else:
                    logger.warning(f"Unknown behaviour label: {label}")

            if not valid:
                valid = [{'label': 'normal_browsing', 'confidence': 0.5}]

            # Sort by confidence
            valid.sort(key=lambda x: x.get('confidence', 0), reverse=True)

            return {
                'behaviours': valid,
                'primary_behaviour': valid[0]['label'],
                'confidence': float(valid[0].get('confidence', 0.5)),
                'reasoning': result.get('reasoning', ''),
            }
        except json.JSONDecodeError:
            logger.warning(f"Parse error: {text[:100]}")
            return {
                'behaviours': [{'label': 'normal_browsing', 'confidence': 0.5}],
                'primary_behaviour': 'normal_browsing',
                'confidence': 0.5,
                'reasoning': 'Parse error',
            }

    def _mock_label(self, cluster_summary):
        """Rule-based mock using DATA-DRIVEN signatures.

        From cluster analysis (Fisher discriminability):
          1. TCP_FLAGS: Impact=2.2, Recon=21.3, Benign=18.7
          2. direction_ratio: Impact=976.8, Benign=167.4
          3. fan_out: Exfil=49.2, Benign=39.1, Impact=9.5
          4. MIN_TTL: Impact=61.3, Recon=27.2, Benign=16.3
          5. OUT_BYTES: Exfil=728310, Benign=54507, Recon=84, Impact=3
        """
        stats = cluster_summary.get('stats', {})
        behaviour = stats.get('behaviour', {})
        graph = stats.get('graph_properties', {})
        proportion = stats.get('proportion', 0.0)

        tcp_flags = behaviour.get('avg_tcp_flags', 18.0)
        out_bytes = behaviour.get('avg_out_bytes', 0)
        min_ttl = behaviour.get('avg_min_ttl', 30)
        fan_out = graph.get('avg_fan_out', 1)
        fan_in = graph.get('avg_fan_in', 1)
        n_src = graph.get('n_unique_sources', 1)
        n_dst = graph.get('n_unique_destinations', 1)
        dir_ratio = behaviour.get('throughput_ratio', 1.0)
        direction = behaviour.get('traffic_direction', '')
        pps = behaviour.get('packets_per_second', 0)

        behaviours = []

        # ── Decision tree based on data analysis ──

        # STEP 1: Check TCP_FLAGS first (Fisher=1.82, top discriminator)
        if tcp_flags < 4:
            # SYN-only flags → Impact (SYN flooding) or connection attempts
            if dir_ratio > 100:
                # Extreme direction ratio → classic SYN flood
                behaviours.append({'label': 'syn_flooding', 'confidence': 0.90})
            elif n_dst <= 5 and n_src <= 3:
                behaviours.append({'label': 'connection_attempts', 'confidence': 0.80})
            else:
                behaviours.append({'label': 'syn_flooding', 'confidence': 0.80})

        else:
            # Normal TCP flags (>4) → NOT SYN flooding
            # STEP 2: Check OUT_BYTES
            if out_bytes > 200000:
                # Very high outbound bytes → definitely data_transfer
                behaviours.append({'label': 'data_transfer', 'confidence': 0.90})

            elif out_bytes > 10000:
                # Moderate-high outbound bytes → could be benign OR exfil
                # STEP 3: Use MIN_TTL to separate
                # Benign=16.3 (low), Exfil=39.0 (moderate)
                if min_ttl < 30:
                    # Low TTL = benign browsing with moderate data
                    behaviours.append({'label': 'normal_browsing', 'confidence': 0.80})
                elif fan_out > 30:
                    # High fan-out + moderate TTL + high bytes = exfil
                    behaviours.append({'label': 'data_transfer', 'confidence': 0.80})
                else:
                    behaviours.append({'label': 'normal_browsing', 'confidence': 0.70})

            elif out_bytes > 200:
                # Low-moderate bytes
                if min_ttl < 25:
                    behaviours.append({'label': 'normal_browsing', 'confidence': 0.75})
                elif min_ttl > 50:
                    if dir_ratio > 100:
                        behaviours.append({'label': 'volumetric_flooding', 'confidence': 0.75})
                    else:
                        behaviours.append({'label': 'service_probing', 'confidence': 0.65})
                else:
                    # Medium TTL (25-50) → Recon territory
                    if pps > 5000 and n_dst > 5:
                        behaviours.append({'label': 'port_scanning', 'confidence': 0.80})
                    else:
                        behaviours.append({'label': 'service_probing', 'confidence': 0.70})

            else:
                # Very low bytes (<200) → scanning or probing
                if min_ttl < 25:
                    behaviours.append({'label': 'normal_browsing', 'confidence': 0.75})
                elif min_ttl > 50:
                    if dir_ratio > 100:
                        behaviours.append({'label': 'volumetric_flooding', 'confidence': 0.75})
                    else:
                        behaviours.append({'label': 'service_probing', 'confidence': 0.65})
                else:
                    if pps > 5000 and n_dst > 5:
                        behaviours.append({'label': 'port_scanning', 'confidence': 0.80})
                    else:
                        behaviours.append({'label': 'service_probing', 'confidence': 0.70})

        # STEP 4: Check for periodic communication (C2)
        dur = behaviour.get('avg_duration_ms', 0)
        if (dur > 10000 and direction == 'BALANCED' and
                n_dst <= 3 and tcp_flags > 10):
            behaviours.append({'label': 'periodic_communication', 'confidence': 0.70})

        # STEP 5: Large proportion → likely benign (aggressive override)
        if proportion > 0.05 and tcp_flags > 8:
            if not any(b['label'] == 'normal_browsing' for b in behaviours):
                behaviours = [{'label': 'normal_browsing', 'confidence': 0.85}]
        # Also override data_transfer if MIN_TTL is low (benign, not exfil)
        if (any(b['label'] == 'data_transfer' for b in behaviours) and
                min_ttl < 25):
            behaviours = [{'label': 'normal_browsing', 'confidence': 0.80}]

        # Deduplicate
        if not behaviours:
            behaviours = [{'label': 'normal_browsing', 'confidence': 0.50}]

        seen = set()
        unique = []
        for b in behaviours:
            if b['label'] not in seen:
                unique.append(b)
                seen.add(b['label'])
        unique.sort(key=lambda x: x['confidence'], reverse=True)

        reasoning = (f"Mock: TCP_FLAGS={tcp_flags:.1f}, OUT_BYTES={out_bytes:.0f}, "
                     f"MIN_TTL={min_ttl:.0f}, fan_out={fan_out:.1f}, "
                     f"dir_ratio={dir_ratio:.1f}, proportion={proportion:.1%}")

        return {
            'behaviours': unique,
            'primary_behaviour': unique[0]['label'],
            'confidence': unique[0]['confidence'],
            'reasoning': reasoning,
        }


    def label_cluster_soft(self, cluster_summary, cluster_weight):
        """Ask LLM to score ALL behaviours (0-1) for a cluster.

        Returns soft distribution instead of single hard label.
        Falls back to None if LLM unavailable (use SoftClusterScorer instead).
        """
        summary_text = cluster_summary.get('text', '')
        prompt = self._build_soft_prompt(summary_text, cluster_weight)

        if self.client is None:
            return None

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a network traffic behaviour analyst. "
                                "Score how strongly each behaviour pattern matches "
                                "the given cluster. Return scores as JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500,
            )
            return self._parse_soft_response(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"LLM soft scoring error: {e}")
            return None

    def _build_soft_prompt(self, summary_text, cluster_weight):
        return f"""Analyze this network traffic cluster and score EACH behaviour
from 0.0 to 1.0 based on how strongly the cluster matches that pattern.
Scores should sum to approximately 1.0.

BEHAVIOUR DEFINITIONS:
  syn_flooding:          SYN-only packets (TCP_FLAGS<4), zero payload, high direction ratio
  volumetric_flooding:   High packet rate, normal TCP, high TTL (>50), extreme direction ratio
  port_scanning:         Normal TCP, very low bytes (<200), high PPS, many destinations
  service_probing:       Normal TCP, small payload (200-5000 bytes), moderate destinations
  data_transfer:         Normal TCP, very high outbound bytes (>50K), moderate-high TTL
  periodic_communication: Balanced bidirectional traffic, few peers, long duration, low PPS
  normal_browsing:       Normal TCP, low TTL (<25), moderate bytes, ephemeral ports
  connection_attempts:   SYN-only to few destinations, zero payload, small scale

KEY THRESHOLDS (from dataset analysis):
  MIN_TTL:   Benign=16, Recon=27, Exfil=39, Impact=61
  TCP_FLAGS: Impact=2 (SYN-only), Exfil=16, Benign=19, Recon=21
  OUT_BYTES: Impact=3, Recon=84, Benign=54507, Exfil=728310

=== CLUSTER TO ANALYZE ===

{summary_text}

Cluster weight: {cluster_weight:.4f}

Respond ONLY in JSON:
{{{{
    "behaviour_scores": {{{{
        "syn_flooding": <0.0-1.0>,
        "volumetric_flooding": <0.0-1.0>,
        "port_scanning": <0.0-1.0>,
        "service_probing": <0.0-1.0>,
        "data_transfer": <0.0-1.0>,
        "periodic_communication": <0.0-1.0>,
        "normal_browsing": <0.0-1.0>,
        "connection_attempts": <0.0-1.0>
    }}}},
    "reasoning": "<brief explanation citing key features>"
}}}}"""

    def _parse_soft_response(self, response_text):
        """Parse LLM soft scoring response."""
        text = response_text.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1] if '\n' in text else text[3:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()
        if text.startswith('json'):
            text = text[4:].strip()

        try:
            result = json.loads(text)
            scores = result.get('behaviour_scores', {})

            valid_scores = {}
            for bhv in BEHAVIOUR_LIST:
                val = scores.get(bhv, 0.0)
                valid_scores[bhv] = max(0.0, min(1.0, float(val)))

            total = sum(valid_scores.values())
            if total > 0:
                valid_scores = {k: v / total for k, v in valid_scores.items()}
            else:
                valid_scores = {'normal_browsing': 1.0}

            primary = max(valid_scores, key=valid_scores.get)

            return {
                'behaviour_scores': valid_scores,
                'primary_behaviour': primary,
                'confidence': valid_scores[primary],
                'reasoning': result.get('reasoning', ''),
            }
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Soft parse error: {e}")
            return None

    def label_all_clusters_soft(self, cluster_summaries, cluster_weights):
        """Score all clusters with soft behaviour distributions via LLM."""
        if self.client is None:
            logger.info("LLM unavailable - use SoftClusterScorer for mock soft")
            return None

        labels = {}
        for k, summary in cluster_summaries.items():
            weight = cluster_weights.get(k, 0.0)
            result = self.label_cluster_soft(summary, weight)
            if result:
                labels[k] = result
                top3 = sorted(result['behaviour_scores'].items(),
                              key=lambda x: -x[1])[:3]
                score_str = ', '.join(f"{b}:{s:.2f}" for b, s in top3)
                logger.info(f"Cluster {k} (LLM soft): [{score_str}]")
        return labels if labels else None

    def label_all_clusters(self, cluster_summaries, cluster_weights):
        labels = {}
        for k, summary in cluster_summaries.items():
            weight = cluster_weights.get(k, 0.0)
            result = self.label_cluster(summary, weight)
            labels[k] = result
            bhvs = ', '.join(b['label'] for b in result['behaviours'])
            logger.info(f"Cluster {k}: [{bhvs}] conf={result['confidence']:.2f}")
        return labels