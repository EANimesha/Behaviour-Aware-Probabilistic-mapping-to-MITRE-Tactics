"""
LLM Semantic Labeling Module
=============================

Stage 5 of the training pipeline:
  - Receives cluster summaries from the Cluster Summarization Module
  - Prompts GPT-4 as a cybersecurity expert
  - Produces structured output: tactic, tactic_id, confidence (P_LLM), reasoning
  - Populates the Knowledge Base

Also used during inference for selective validation and during
streaming for re-labeling when drift or new clusters are detected.
"""

import json
import os
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMLabeler:
    """LLM-based cluster labeler using GPT-4.

    Produces structured tactic assignments for the Knowledge Base.
    """

    def __init__(self, api_key=None, model='gpt-4'):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.client = None

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"LLMLabeler initialized with model={model}")
        except ImportError:
            logger.warning("OpenAI library not installed. Using mock labeler.")
        except Exception as e:
            logger.warning(f"OpenAI init failed: {e}. Using mock labeler.")

    def label_cluster(self, cluster_summary, cluster_weight):
        """Label a single cluster using LLM reasoning.

        Args:
            cluster_summary: dict with 'text' and 'stats' from ClusterSummarizer
            cluster_weight: float, DP-GMM weight for this cluster (P_GMM prior)

        Returns:
            dict with: tactic, tactic_id, p_llm, reasoning
        """
        summary_text = cluster_summary['text']
        prompt = self._build_labeling_prompt(summary_text, cluster_weight)

        if self.client is None:
            # return self._mock_label(cluster_summary)
            print(cluster_summary)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a cybersecurity expert specializing in "
                                "MITRE ATT&CK framework analysis of network traffic."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400,
            )

            response_text = response.choices[0].message.content
            result = self._parse_response(response_text)
            return result

        except Exception as e:
            logger.error(f"LLM API error: {e}")
            # return self._mock_label(cluster_summary)

    def _build_labeling_prompt(self, summary_text, cluster_weight):
        """Build the structured prompt for cluster labeling.

        Gives GPT-4 specific network-level indicators for each tactic
        so it can match cluster statistics to attack patterns.
        """

        return f"""
        You are a cybersecurity expert specializing in network traffic analysis.

        Analyze the following network traffic cluster summary and estimate how strongly this cluster matches each relevant MITRE ATT&CK tactic.

        {summary_text}

        Cluster prior weight from DP-GMM: {cluster_weight:.4f}

        A cluster may represent more than one tactic. Do NOT force it into only one tactic.
        Instead, provide a probability distribution over the most relevant tactics. Each probability must be written with exactly 3 decimal places.

        Use the following network-level indicators as guidance:

        - Reconnaissance (TA0043): many unique destinations from few sources, port scanning, very short flows, small packets, high fan-out, low bytes per flow, common destination ports.

        - Initial Access (TA0001): targeting exposed services such as 80, 443, 22, or 445, connection attempts, moderate duration, exploit-like or fuzzing traffic, repeated attempts to few destinations.

        - Execution (TA0002): payload delivery patterns, moderate or large packets, unusual TCP flags, short-to-medium duration, one-to-one or one-to-few communication.

        - Persistence (TA0003): long-duration flows, periodic or repeated traffic, low packet rate, one-to-one communication, balanced or inbound-heavy traffic.

        - Discovery (TA0007): probing behavior, ICMP or multiple protocols, moderate fan-out, short-to-medium flows, internal service discovery patterns.

        - Lateral Movement (TA0008): internal source and destination communication, multiple internal pairs, TCP-dominant traffic, spreading behavior, use of service ports or ephemeral ports.

        - Command and Control (TA0011): long or repeated connections, few unique IPs, balanced bidirectional traffic, steady byte rate, DNS/HTTP-like communication, low flag variability.

        - Exfiltration (TA0010): outbound-heavy traffic, large source-to-destination bytes, large packets, moderate-to-long duration, high upload ratio.

        - Impact (TA0040): DoS/DDoS patterns, very high packet count, high throughput, many-to-one or one-to-many traffic, SYN-heavy behavior, short intense flows.

        - Benign: large cluster weight, values close to global averages, balanced traffic, normal ports, typical TTL values, no extreme behavior.

        Important rules:
        1. Use the actual numeric values in the summary, such as bytes, packets, duration, ports, unique sources/destinations, and ratios.
        2. Assign probabilities to multiple tactics when the evidence overlaps.
        3. Probabilities must sum to 1.0.
        4. Include only tactics with meaningful evidence. Do not include every tactic unless needed.
        5. If the cluster is unclear, distribute probability across possible tactics and explain the uncertainty.
        6. Treat the DP-GMM cluster prior weight as supporting evidence only, not the main deciding factor.

        Respond ONLY in JSON format:

        {{
          "primary_tactic": "<most likely tactic name>",
          "primary_tactic_id": "<TA#### or Benign>",
          "tactic_probabilities": [
            {{
              "tactic": "<tactic name>",
              "tactic_id": "<TA#### or Benign>",
              "probability": <0.0-1.0>,
              "evidence": "<short evidence using specific numeric values from the summary>"
            }}
          ],
          "confidence": <0.0-1.0>,
          "reasoning": "<2-4 sentences explaining why the probabilities were assigned, citing specific numeric values from the summary>"
        }}
        """

        #
        # return f"""You are a cybersecurity expert specializing in network traffic analysis.
        # Analyze the following network traffic cluster summary and assign the most
        # likely MITRE ATT&CK tactic based on the concrete network-level indicators.
        #
        # {summary_text}
        #
        # Cluster prior weight (from DP-GMM): {cluster_weight:.4f}
        #
        # Use these network-level indicators to determine the tactic:
        #
        # - Reconnaissance (TA0043): Many unique destinations from few sources (port scanning),
        #   very short flows (<10ms), small packets, high fan-out, destination ports targeting
        #   well-known range (0-1023), low bytes per flow
        #
        # - Initial Access (TA0001): Targeting well-known dst ports (80, 443, 22, 445),
        #   exploit payloads = moderate-large packets, TCP flags showing connection attempts,
        #   moderate duration, fuzzing = high packet count to few destinations
        #
        # - Execution (TA0002): Large payload packets (shellcode delivery), unusual TCP flags,
        #   short-medium duration, one-to-one or one-to-few connectivity
        #
        # - Persistence (TA0003): Long-duration flows, one-to-one connectivity, periodic traffic
        #   patterns, low packet rate, backdoor = inbound-heavy or balanced traffic
        #
        # - Discovery (TA0007): ICMP prevalent, moderate fan-out, probing multiple protocols,
        #   balanced traffic, short-medium duration
        #
        # - Lateral Movement (TA0008): Internal IP ranges, multiple source-destination pairs,
        #   ephemeral source ports, TCP-dominant, spreading pattern
        #
        # - Command and Control (TA0011): Long duration, one-to-one connectivity, very few
        #   unique IPs (1-3 each side), balanced bidirectional traffic, low TCP flag variability,
        #   moderate steady byte rate, DNS or HTTP app protocol
        #
        # - Exfiltration (TA0010): Outbound-heavy traffic (high src-to-dst throughput ratio),
        #   large outbound bytes, moderate-long duration, large packets
        #
        # - Impact (TA0040): DoS/DDoS = many-to-one or one-to-many, very high packet counts,
        #   high throughput, short duration, SYN-heavy TCP flags, very high fan-in
        #
        # - Benign: Large proportion of total traffic (>20%), values close to global averages,
        #   balanced traffic, normal port distribution, typical TTL values, no extreme features
        #
        # IMPORTANT: Use the actual numeric values provided (bytes, packets, milliseconds)
        # to make your assessment. A cluster with 50,000 bytes average is very different
        # from one with 50 bytes. Duration of 5ms is scanning; duration of 30,000ms
        # is a persistent connection.
        #
        # Respond ONLY in JSON format:
        # {{
        #     "tactic": "<tactic name>",
        #     "tactic_id": "<TA####>",
        #     "technique_id": "<T####>",
        #     "technique_name": "<technique name>",
        #     "confidence": <0.0-1.0>,
        #     "reasoning": "<2-3 sentences citing specific numeric values from the summary>"
        # }}"""

#         return f"""You are a cybersecurity expert specializing in network traffic analysis.
# Analyze the following network traffic cluster summary and assign the most
# likely MITRE ATT&CK tactic based on the concrete network-level indicators.
#
# {summary_text}
#
# Cluster prior weight (from DP-GMM): {cluster_weight:.4f}
#
# Use these network-level indicators to determine the tactic:
#
# - Reconnaissance (TA0043): Many unique destinations from few sources (port scanning),
#   very short flows (<10ms), small packets, high fan-out, destination ports targeting
#   well-known range (0-1023), low bytes per flow
#
# - Initial Access (TA0001): Targeting well-known dst ports (80, 443, 22, 445),
#   exploit payloads = moderate-large packets, TCP flags showing connection attempts,
#   moderate duration, fuzzing = high packet count to few destinations
#
# - Execution (TA0002): Large payload packets (shellcode delivery), unusual TCP flags,
#   short-medium duration, one-to-one or one-to-few connectivity
#
# - Persistence (TA0003): Long-duration flows, one-to-one connectivity, periodic traffic
#   patterns, low packet rate, backdoor = inbound-heavy or balanced traffic
#
# - Discovery (TA0007): ICMP prevalent, moderate fan-out, probing multiple protocols,
#   balanced traffic, short-medium duration
#
# - Lateral Movement (TA0008): Internal IP ranges, multiple source-destination pairs,
#   ephemeral source ports, TCP-dominant, spreading pattern
#
# - Command and Control (TA0011): Long duration, one-to-one connectivity, very few
#   unique IPs (1-3 each side), balanced bidirectional traffic, low TCP flag variability,
#   moderate steady byte rate, DNS or HTTP app protocol
#
# - Exfiltration (TA0010): Outbound-heavy traffic (high src-to-dst throughput ratio),
#   large outbound bytes, moderate-long duration, large packets
#
# - Impact (TA0040): DoS/DDoS = many-to-one or one-to-many, very high packet counts,
#   high throughput, short duration, SYN-heavy TCP flags, very high fan-in
#
# - Benign: Large proportion of total traffic (>20%), values close to global averages,
#   balanced traffic, normal port distribution, typical TTL values, no extreme features
#
# IMPORTANT: Use the actual numeric values provided (bytes, packets, milliseconds)
# to make your assessment. A cluster with 50,000 bytes average is very different
# from one with 50 bytes. Duration of 5ms is scanning; duration of 30,000ms
# is a persistent connection.
#
# Respond ONLY in JSON format:
# {{
#     "tactic": "<tactic name>",
#     "tactic_id": "<TA####>",
#     "confidence": <0.0-1.0>,
#     "reasoning": "<2-3 sentences citing specific numeric values from the summary>"
# }}"""


    #     return f"""You are a cybersecurity expert. Analyze the following network traffic
    # cluster and assign the most likely MITRE ATT&CK tactic.
    #
    # {summary_text}
    #
    # Cluster prior weight (from DP-GMM): {cluster_weight:.4f}
    #
    # Based on the feature patterns, behavioural signature, and network topology,
    # assign a MITRE ATT&CK tactic. Consider these tactics:
    # - Reconnaissance (TA0043): scanning, probing
    # - Resource Development (TA0042): infrastructure setup
    # - Initial Access (TA0001): exploitation, phishing
    # - Execution (TA0002): code execution
    # - Persistence (TA0003): maintaining access
    # - Privilege Escalation (TA0004): gaining higher privileges
    # - Defense Evasion (TA0005): avoiding detection
    # - Credential Access (TA0006): stealing credentials
    # - Discovery (TA0007): exploring the environment
    # - Lateral Movement (TA0008): moving through the network
    # - Collection (TA0009): gathering data
    # - Command and Control (TA0011): C2 communication
    # - Exfiltration (TA0010): stealing data
    # - Impact (TA0040): disruption, DoS
    # - Benign: normal traffic
    #
    # Respond ONLY in JSON format:
    # {{
    #     "tactic": "<tactic name>",
    #     "tactic_id": "<TA####>",
    #     "confidence": <0.0-1.0>,
    #     "reasoning": "<2-3 sentence explanation of why this tactic fits>"
    # }}"""

    def _parse_response(self, response_text):
        """Parse JSON response from LLM."""
        # Strip markdown code fences if present
        text = response_text.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1] if '\n' in text else text[3:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()
        if text.startswith('json'):
            text = text[4:].strip()

        # try:
        #     result = json.loads(text)
        #     return {
        #         'tactic': result.get('tactic', 'Unknown'),
        #         'tactic_id': result.get('tactic_id', 'Unknown'),
        #         # 'technique_id': result.get('technique_id', 'Unknown'),
        #         # 'technique_name': result.get('technique_name', 'Unknown'),
        #         'p_llm': float(result.get('confidence', 0.5)),
        #         'reasoning': result.get('reasoning', ''),
        #     }
        # except json.JSONDecodeError:
        #     logger.warning(f"Could not parse LLM response: {text[:100]}...")
        #     return {
        #         'tactic': 'Unknown',
        #         'tactic_id': 'Unknown',
        #         # 'technique_id': 'Unknown',
        #         # 'technique_name': 'Unknown',
        #         'p_llm': 0.5,
        #         'reasoning': 'Parse error',
        #     }
        try:
            result = json.loads(text)

            tactic_probabilities = result.get("tactic_probabilities", [])

            # Validate and clean probability list
            cleaned_probs = []
            for item in tactic_probabilities:
                cleaned_probs.append({
                    "tactic": item.get("tactic", "Unknown"),
                    "tactic_id": item.get("tactic_id", "Unknown"),
                    "probability": float(item.get("probability", 0.0)),
                    "evidence": item.get("evidence", "")
                })

            return {
                "primary_tactic": result.get("primary_tactic", "Unknown"),
                "primary_tactic_id": result.get("primary_tactic_id", "Unknown"),
                "tactic_probabilities": cleaned_probs,
                "p_llm": float(result.get("confidence", 0.5)),
                "reasoning": result.get("reasoning", "")
            }

        except json.JSONDecodeError:
            logger.warning(f"Could not parse LLM response: {text[:100]}...")

            return {
                "primary_tactic": "Unknown",
                "primary_tactic_id": "Unknown",
                "tactic_probabilities": [],
                "p_llm": 0.5,
                "reasoning": "Parse error"
            }

    def label_all_clusters(self, cluster_summaries, cluster_weights):
        """Label all clusters and return structured results.

        Args:
            cluster_summaries: dict of {k: summary_dict} from ClusterSummarizer
            cluster_weights: dict of {k: weight} from DP-GMM

        Returns:
            labels: dict of {k: label_result_dict}
        """
        labels = {}
        for k, summary in cluster_summaries.items():
            weight = cluster_weights.get(k, 0.0)
            result = self.label_cluster(summary, weight)
            labels[k] = result
            logger.info(f"Cluster {k}: {result['primary_tactic']} "
                         f"(P_LLM={result['p_llm']:.2f})")
        return labels