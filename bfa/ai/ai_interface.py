"""
AI Interface module for BattleFieldAgents game.
Handles communication with the AI API to get agent decisions.
"""

from bfa.core.constants import *
from bfa.utils.utils import format_agent_state, get_possible_moves, distance, has_line_of_sight
import requests
import json
import random
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AIInterface:
    """
    Interface for communicating with the AI API.
    Sends game state and receives agent decisions (thoughts + actions).
    """
    
    def __init__(self, api_url="https://unpalpablely-vibronic-leonore.ngrok-free.dev/api/v1", timeout=API_TIMEOUT):
        """
        Initialize the AI interface.
        
        Args:
            api_url (str): URL of the AI API endpoint.
            timeout (float): Request timeout in seconds.
        """
        # Ensure the URL points to the chat completions endpoint if it's an OpenAI-compatible API
        if not api_url.endswith("/chat/completions"):
             self.api_url = api_url.rstrip("/") + "/chat/completions"
        else:
             self.api_url = api_url

        self.timeout = timeout
        self.api_key = os.getenv("API_KEY")
        self.last_response = None
        self.is_thinking = False
        
        # Load system message
        try:
            with open('bfa/ai/system_message.txt', 'r') as f:
                self.system_message = f.read()
        except FileNotFoundError:
            print("Error: system_message.txt not found.")
            self.system_message = "You are an AI agent playing a game."
    
    def get_agent_decision(self, agent, turn, game_state):
        """
        Request a decision from the AI for a specific agent.
        
        Args:
            agent (Agent): The agent that needs to make a decision
            turn (dict): Current turn information
            game_state: The game state object
        
        Returns:
            tuple: (thoughts, action) where:
                - thoughts (str): The agent's reasoning
                - action (str): The action string (e.g., "MOVE [3, 5]")
            Returns (None, None) if the request fails.
        """
        self.is_thinking = True
        
        try:
            # Format the agent's state for the API
            state = format_agent_state(
                agent,
                turn,
                game_state.agents,
                game_state.targets,
                game_state.obstacles
            )
            
            # Prepare the request payload for OpenAI-compatible API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "qwen/qwen3-30b-a3b-2507:2", 
                "messages": [
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": json.dumps(state)}
                ],
                "temperature": 0.2
            }

            # Send POST request to the AI API
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
                headers=headers
            )
            
            # Check if request was successful
            if response.status_code != 200:
                print(f"AI API Error: Status code {response.status_code} - {response.text}")
                self.is_thinking = False
                return None, None
            
            # Parse the response
            data = response.json()
            
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0]['message']['content']
                thoughts, action = self._parse_response(content)
                
                self.last_response = data
                self.is_thinking = False
                return thoughts, action
            else:
                print("AI API Error: Unexpected response format")
                self.is_thinking = False
                return None, None
        
        except requests.exceptions.Timeout:
            print("AI API Error: Request timed out")
            self.is_thinking = False
            return None, None
        
        except requests.exceptions.ConnectionError:
            print("AI API Error: Could not connect to server")
            print(f"Make sure the API is running at {self.api_url}")
            self.is_thinking = False
            return None, None
        
        except requests.exceptions.RequestException as e:
            print(f"AI API Error: {e}")
            self.is_thinking = False
            return None, None
        
        except json.JSONDecodeError:
            print("AI API Error: Invalid JSON response")
            self.is_thinking = False
            return None, None
        
        except Exception as e:
            print(f"AI API Error: Unexpected error - {e}")
            self.is_thinking = False
            return None, None

    def _parse_response(self, content):
        """
        Parse the content string to extract thoughts and action.
        """
        thoughts = ""
        action = ""
        try:
            lines = content.split('\n')
            # Extract thoughts
            t_lines = [l for l in lines if l.startswith('THOUGHTS: ')]
            if t_lines:
                thoughts = t_lines[0][10:].strip()
            
            # Extract action
            a_lines = [l for l in lines if l.startswith('ACTION: ')]
            if a_lines:
                action = a_lines[0][8:].strip()
            
            return thoughts, action
        except Exception as e:
            print(f"Error parsing response: {e}")
            return "", ""
    
    def check_api_connection(self):
        """
        Check if the API is reachable.
        
        Returns:
            bool: True if API is accessible, False otherwise
        """
        try:
            # Try to connect to a models endpoint or similar to check availability
            # Since we don't know if /models is available on the private API, 
            # we'll assume it's up if we can reach the base URL or just skip this check strictly.
            # But for good measure let's try a simple GET to the base URL
            test_url = self.api_url.replace('/chat/completions', '')
            response = requests.get(test_url, timeout=5)
            # Accept any response that indicates the server is there (even 404/401 is better than connection error)
            return True
        except:
            return False


class MockAIInterface(AIInterface):
    """
    Mock AI interface for testing without an actual API.
    Provides simple rule-based decisions for agents.
    """
    
    def __init__(self):
        """Initialize the mock AI interface."""
        # Initialize parent but don't fail if files missing
        self.api_url = "MOCK"
        self.timeout = 0
        self.api_key = "MOCK"
        self.system_message = ""
        self.last_response = None
        self.is_thinking = False
    
    def get_agent_decision(self, agent, turn, game_state):
        """
        Generate a mock decision for an agent.
        - Priority 1: Attack visible enemies with clear Line of Sight (LOS).
        - Priority 2: Move towards the enemy main target.
        - Priority 3: Wait.
        """
        self.is_thinking = True
        
        # 1. Check for attack opportunities first
        visible_enemies = [
            e for e in agent.sight
            if e.get('kind') in ['agents', 'targets'] and e.get('team') != agent.team
        ]
        
        if visible_enemies:
            # Find the closest enemy
            closest_enemy = min(
                visible_enemies,
                key=lambda e: distance(agent.position, e['position'])
            )
            enemy_pos = closest_enemy['position']
            
            thoughts = f"Enemy '{closest_enemy.get('id', 'target')}' spotted at {enemy_pos}."
            
            # Check for a clear Line of Sight (LOS)
            if has_line_of_sight(agent.position, enemy_pos, game_state.agents, game_state.targets, game_state.obstacles):
                action = f"ATTACK [{enemy_pos[0]}, {enemy_pos[1]}]"
                thoughts += " Clear line of sight. Attacking!"
                self.is_thinking = False
                return thoughts, action
            else:
                thoughts += " No clear line of sight."

        # 2. No enemy with LOS, so move towards the main enemy target
        enemy_target = next((t for t in game_state.targets if t.team != agent.team and t.is_alive()), None)
        
        if enemy_target:
            possible_moves = get_possible_moves(
                agent,
                game_state.agents,
                game_state.targets,
                game_state.obstacles
            )
            
            if possible_moves:
                # Find the move that gets closest to the enemy target
                best_move = min(
                    possible_moves,
                    key=lambda move: distance(move, enemy_target.position)
                )
                
                thoughts = f"No enemy in my line of sight. Moving towards the enemy target at {enemy_target.position}."
                action = f"MOVE [{best_move[0]}, {best_move[1]}]"
                self.is_thinking = False
                return thoughts, action

        # 3. If no other action, wait
        thoughts = "No valid moves or attacks available. Waiting."
        action = "WAIT"
        self.is_thinking = False
        return thoughts, action
    
    def check_api_connection(self):
        """Mock API is always 'connected'."""
        return True

class MockAIInterface(AIInterface):
    """
    Mock AI interface for testing without an actual API.
    Provides simple rule-based decisions for agents.
    """
    
    def __init__(self):
        """Initialize the mock AI interface."""
        # Initialize parent but don't fail if files missing
        self.api_url = "MOCK"
        self.timeout = 0
        self.api_key = "MOCK"
        self.system_message = ""
        self.last_response = None
        self.is_thinking = False
    
    def get_agent_decision(self, agent, turn, game_state):
        """
        Generate a mock decision for an agent.
        - Priority 1: Attack visible enemies with clear Line of Sight (LOS).
        - Priority 2: Move towards the enemy main target.
        - Priority 3: Wait.
        """
        self.is_thinking = True
        
        # 1. Check for attack opportunities first
        visible_enemies = [
            e for e in agent.sight
            if e.get('kind') in ['agents', 'targets'] and e.get('team') != agent.team
        ]
        
        if visible_enemies:
            # Find the closest enemy
            closest_enemy = min(
                visible_enemies,
                key=lambda e: distance(agent.position, e['position'])
            )
            enemy_pos = closest_enemy['position']
            
            thoughts = f"Enemy '{closest_enemy.get('id', 'target')}' spotted at {enemy_pos}."
            
            # Check for a clear Line of Sight (LOS)
            if has_line_of_sight(agent.position, enemy_pos, game_state.agents, game_state.targets, game_state.obstacles):
                action = f"ATTACK [{enemy_pos[0]}, {enemy_pos[1]}]"
                thoughts += " Clear line of sight. Attacking!"
                self.is_thinking = False
                return thoughts, action
            else:
                thoughts += " No clear line of sight."

        # 2. No enemy with LOS, so move towards the main enemy target
        enemy_target = next((t for t in game_state.targets if t.team != agent.team and t.is_alive()), None)
        
        if enemy_target:
            possible_moves = get_possible_moves(
                agent,
                game_state.agents,
                game_state.targets,
                game_state.obstacles
            )
            
            if possible_moves:
                # Find the move that gets closest to the enemy target
                best_move = min(
                    possible_moves,
                    key=lambda move: distance(move, enemy_target.position)
                )
                
                thoughts = f"No enemy in my line of sight. Moving towards the enemy target at {enemy_target.position}."
                action = f"MOVE [{best_move[0]}, {best_move[1]}]"
                self.is_thinking = False
                return thoughts, action

        # 3. If no other action, wait
        thoughts = "No valid moves or attacks available. Waiting."
        action = "WAIT"
        self.is_thinking = False
        return thoughts, action
    
    def check_api_connection(self):
        """Mock API is always 'connected'."""
        return True

class BenAI(AIInterface):
    """
    AI interface for grouped agents that stay together and shoot enemies.
    
    Strategy:
    - Priority 1: Attack visible enemies with clear Line of Sight (LOS)
    - Priority 2: Move to rejoin the group if separated
    - Priority 3: Move as a group towards the enemy target
    - Priority 4: Wait if grouped and no target visible
    
    Group formation: Agents stay within a proximity radius of their teammates.
    """
    
    # Formation parameters
    GROUP_PROXIMITY_RADIUS = 5  # Maximum distance to stay grouped
    GROUP_CENTER_SEARCH_RADIUS = 8  # Radius to look for group center
    
    def __init__(self):
        """Initialize the BenAI interface."""
        self.api_url = "MOCK"
        self.timeout = 0
        self.api_key = "MOCK"
        self.system_message = ""
        self.last_response = None
        self.is_thinking = False
    
    def _get_team_agents(self, agent, game_state):
        """
        Get all alive teammates.
        
        Args:
            agent: Current agent
            game_state: Game state
            
        Returns:
            list: List of teammate agents
        """
        return [a for a in game_state.agents 
                if a.team == agent.team and a.is_alive() and a.id != agent.id]
    
    def _calculate_group_center(self, agent, game_state):
        """
        Calculate the center position of the agent's team.
        
        Args:
            agent: Current agent
            game_state: Game state
            
        Returns:
            list: Average position [x, y] of all teammates
        """
        teammates = self._get_team_agents(agent, game_state)
        
        if not teammates:
            return agent.position
        
        all_positions = [agent.position] + [t.position for t in teammates]
        avg_x = sum(p[0] for p in all_positions) / len(all_positions)
        avg_y = sum(p[1] for p in all_positions) / len(all_positions)
        
        return [int(avg_x), int(avg_y)]
    
    def _get_visible_enemies(self, agent):
        """
        Get all visible enemies from agent's sight.
        
        Args:
            agent: Current agent
            
        Returns:
            list: List of visible enemy entities
        """
        return [e for e in agent.sight
                if e.get('kind') in ['agents', 'targets'] and e.get('team') != agent.team]
    
    def _is_grouped(self, agent, game_state):
        """
        Check if agent is close enough to the group center.
        
        Args:
            agent: Current agent
            game_state: Game state
            
        Returns:
            bool: True if agent is within GROUP_PROXIMITY_RADIUS of group center
        """
        group_center = self._calculate_group_center(agent, game_state)
        dist_to_center = distance(agent.position, group_center)
        return dist_to_center <= self.GROUP_PROXIMITY_RADIUS
    
    def get_agent_decision(self, agent, turn, game_state):
        """
        Generate a decision for an agent following group formation strategy.
        
        Args:
            agent: Current agent
            turn: Current turn information
            game_state: Game state
            
        Returns:
            tuple: (thoughts, action)
        """
        self.is_thinking = True
        
        visible_enemies = self._get_visible_enemies(agent)
        group_center = self._calculate_group_center(agent, game_state)
        is_grouped = self._is_grouped(agent, game_state)
        
        # PRIORITY 1: Attack visible enemies with clear Line of Sight
        if visible_enemies:
            # Find the closest enemy
            closest_enemy = min(
                visible_enemies,
                key=lambda e: distance(agent.position, e['position'])
            )
            enemy_pos = closest_enemy['position']
            enemy_id = closest_enemy.get('id', 'target')
            
            # Check for a clear Line of Sight (LOS)
            if has_line_of_sight(agent.position, enemy_pos, game_state.agents, game_state.targets, game_state.obstacles):
                action = f"ATTACK [{enemy_pos[0]}, {enemy_pos[1]}]"
                thoughts = f"Enemy '{enemy_id}' spotted! GROUPED ATTACK at {enemy_pos}!"
                self.is_thinking = False
                return thoughts, action
            else:
                # Can see enemy but no LOS - try to move to clear line of sight
                thoughts = f"Enemy '{enemy_id}' spotted but no clear line of sight. "
        
        # PRIORITY 2: If not grouped, rejoin the group
        if not is_grouped:
            possible_moves = get_possible_moves(
                agent,
                game_state.agents,
                game_state.targets,
                game_state.obstacles
            )
            
            if possible_moves:
                # Find move that gets closest to group center
                best_move = min(
                    possible_moves,
                    key=lambda move: distance(move, group_center)
                )
                
                thoughts = f"Rejoining group at {group_center}. Moving to {best_move}."
                action = f"MOVE [{best_move[0]}, {best_move[1]}]"
                self.is_thinking = False
                return thoughts, action
        
        # PRIORITY 3: Move as a group towards enemy target
        enemy_target = next((t for t in game_state.targets if t.team != agent.team and t.is_alive()), None)
        
        if enemy_target:
            possible_moves = get_possible_moves(
                agent,
                game_state.agents,
                game_state.targets,
                game_state.obstacles
            )
            
            if possible_moves:
                # Find move that both gets close to group and towards enemy
                best_move = min(
                    possible_moves,
                    key=lambda move: (
                        distance(move, group_center) * 0.5 +  # Stay grouped (weight: 0.5)
                        distance(move, enemy_target.position) * 0.5  # Move towards target (weight: 0.5)
                    )
                )
                
                thoughts = f"Moving with group towards enemy at {enemy_target.position}."
                action = f"MOVE [{best_move[0]}, {best_move[1]}]"
                self.is_thinking = False
                return thoughts, action
        
        # PRIORITY 4: Stay grouped and wait
        thoughts = "Grouped and holding position. Waiting for enemy."
        action = "WAIT"
        self.is_thinking = False
        return thoughts, action
    
    def check_api_connection(self):
        """BenAI is always 'connected'."""
        return True


class ExplorerAI(AIInterface):
    """
    Aggressive team strategy:
    All 3 agents swarm towards the nearest enemy in coordinated attack.
    
    Decision logic:
    1. If enemy visible with LOS → ATTACK immediately
    2. Else: All move towards nearest enemy (agent or base)
    3. Stay loosely grouped while pushing forward
    """
    
    def __init__(self):
        """Initialize the ExplorerAI interface."""
        self.api_url = "MOCK"
        self.timeout = 0
        self.api_key = "MOCK"
        self.system_message = ""
        self.last_response = None
        self.is_thinking = False
    
    def _get_visible_enemies(self, agent):
        """Get all visible enemies."""
        return [e for e in agent.sight
                if e.get('kind') in ['agents', 'targets'] and e.get('team') != agent.team]
    
    def _get_all_enemy_agents(self, agent, game_state):
        """Get ALL enemy agents (alive)."""
        return [a for a in game_state.agents 
                if a.team != agent.team and a.is_alive()]
    
    def _get_enemy_base(self, team, game_state):
        """Get the enemy base."""
        return next((t for t in game_state.targets if t.team != team and t.is_alive()), None)
    
    def _get_nearest_enemy(self, agent, game_state):
        """
        Get the nearest enemy (agent or base) from agent's position.
        Returns the closest target.
        """
        all_enemy_agents = self._get_all_enemy_agents(agent, game_state)
        enemy_base = self._get_enemy_base(agent.team, game_state)
        
        nearest_target = None
        nearest_dist = float('inf')
        
        # Check all enemy agents
        for enemy_agent in all_enemy_agents:
            dist = distance(agent.position, enemy_agent.position)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_target = enemy_agent
        
        # Check enemy base
        if enemy_base:
            dist = distance(agent.position, enemy_base.position)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_target = enemy_base
        
        return nearest_target
    
    def get_agent_decision(self, agent, turn, game_state):
        """
        Aggressive swarm strategy: All agents attack nearest enemy.
        
        Args:
            agent: Current agent
            turn: Turn information
            game_state: Game state
            
        Returns:
            tuple: (thoughts, action)
        """
        self.is_thinking = True
        
        visible_enemies = self._get_visible_enemies(agent)
        nearest_target = self._get_nearest_enemy(agent, game_state)
        
        possible_moves = get_possible_moves(
            agent, game_state.agents, game_state.targets, game_state.obstacles
        )
        
        if not possible_moves:
            thoughts = "No valid moves. Waiting."
            action = "WAIT"
            self.is_thinking = False
            return thoughts, action
        
        # PRIORITY 1: ATTACK visible enemy with LOS
        if visible_enemies:
            closest_visible = min(visible_enemies, key=lambda e: distance(agent.position, e['position']))
            enemy_pos = closest_visible['position']
            
            if has_line_of_sight(agent.position, enemy_pos, game_state.agents, game_state.targets, game_state.obstacles):
                action = f"ATTACK [{enemy_pos[0]}, {enemy_pos[1]}]"
                enemy_id = closest_visible.get('id', 'target')
                thoughts = f"FIRE! Attacking {enemy_id} at {enemy_pos}!"
                self.is_thinking = False
                return thoughts, action
        
        # PRIORITY 2: Swarm towards nearest enemy
        if nearest_target:
            best_move = min(possible_moves, key=lambda m: distance(m, nearest_target.position))
            
            target_desc = f"{nearest_target.id}" if hasattr(nearest_target, 'id') else "enemy base"
            target_dist = distance(agent.position, nearest_target.position)
            thoughts = f"AGGRESSION! Charging {target_desc} ({int(target_dist)} cells away)!"
            action = f"MOVE [{best_move[0]}, {best_move[1]}]"
            self.is_thinking = False
            return thoughts, action
        
        # Fallback
        thoughts = "No enemies. Moving forward."
        action = f"MOVE [{possible_moves[0][0]}, {possible_moves[0][1]}]"
        self.is_thinking = False
        return thoughts, action
    
    def check_api_connection(self):
        """ExplorerAI is always 'connected'."""
        return True


# Example usage
if __name__ == "__main__":
    # Test mock interface
    print("Testing mock interface...")
    mock_ai = MockAIInterface()
    print(f"✓ Mock AI created (always returns valid decisions)")

