// Connect4.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include "../NNet/neural_net.h"

const double INF = 1e18;

class Error {
public:
    Error(std::ostream& display, const std::string& msg) {
        display << msg;
    }
};

class Connect4 {
private:
    bool turn = false;
    int state = 2;

    int rows = 6, cols = 7;
    std::vector<int> board = std::vector<int>{};

public:
    Connect4() {
        for (int i = 0; i < rows * cols; i++) board.push_back(0);
    }
    Connect4(const Connect4& org) {
        state = org.State();
        turn = org.Turn();
        SetPosition(org.GetPosition());
    }

    void Reset() {
        for (int i = 0; i < rows * cols; i++) board[i] = 0;
        turn = false;
        state = 2;
    }

    std::vector<int> GetPosition() const { return board; }
    void SetPosition(const std::vector<int>& pos) {
        board = pos;
        CheckPosition();

        int rcnt = 0, ycnt = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (board[i * cols + j] == 1) rcnt++;
                else if (board[i * cols + j] == -1) ycnt++;
            }
        }

        if (rcnt == ycnt) turn = false;
        else turn = true;
    }

    bool Turn() const { return turn; }
    int State() const { return state; }

    void Move(int col) {
        if (state != 2) throw Error{ std::cout, "Connect4 : game is already finished!\n" };

        if (col < 0 || col >= cols || board[0 * cols + col]) {
            throw Error{ std::cout, "Connect4 : invalid move!\n" };
            return;
        }

        for (int i = rows - 1; i >= 0; i--) {
            if (board[i * cols + col] == 0) {
                board[i * cols + col] = turn ? -1 : 1;
                break;
            }
        }

        turn = !turn;
        CheckPosition();
    }
    void UndoMove(int col) {
        if (col < 0 || col >= cols) throw Error{ std::cout, "Connect4 : invalid column!\n" };
        if (board[(rows - 1) * cols + col] == 0) throw Error{ std::cout, "Connect4 : cannot undo a move here!\n" };

        for (int i = 0; i < rows; i++) {
            if (board[i * cols + col] != 0) {
                board[i * cols + col] = 0;
                break;
            }
        }

        turn = !turn;
        state = 2;
    }
    std::vector<int> PossibleMoves() const {
        std::vector<int> ret;

        for (int i = 0; i < cols; i++) if (board[0 * cols + i] == 0) ret.push_back(i);

        return ret;
    }

    int CheckPosition() {
        if (state != 2) return state;

        // rowwise
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j <= cols - 4; j++) {
                if (board[i * cols + j] == 0) continue;

                bool good = true;
                for (int k = 1; k < 4; k++) {
                    if (board[i * cols + j + k] != board[i * cols + j]) {
                        good = false;
                        break;
                    }
                }

                if (good) return state = board[i * cols + j];
            }
        }

        // columnwise
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i <= rows - 4; i++) {
                if (board[i * cols + j] == 0) continue;

                bool good = true;
                for (int k = 1; k < 4; k++) {
                    if (board[(i + k) * cols + j] != board[i * cols + j]) {
                        good = false;
                        break;
                    }
                }

                if (good) {
                    return state = board[i * cols + j];
                }
            }
        }

        //diagonal NE-SW

        for (int i = 0; i <= rows - 4; i++) {
            for (int j = 3; j < cols; j++) {
                if (board[i * cols + j] == 0) continue;

                bool good = true;
                for (int d = 1; d < 4; d++) {
                    if (board[(i + d) * cols + j - d] != board[i * cols + j]) {
                        good = false;
                        break;
                    }
                }

                if (good) return state = board[i * cols + j];
            }
        }

        //diagonal NW-SE
        for (int i = 0; i <= rows - 4; i++) {
            for (int j = 0; j <= cols - 4; j++) {
                if (board[i * cols + j] == 0) continue;

                bool good = true;
                for (int d = 1; d < 4; d++) {
                    if (board[(i + d) * cols + j + d] != board[i * cols + j]) {
                        good = false;
                        break;
                    }
                }

                if (good) return state = board[i * cols + j];
            }
        }

        bool draw = true;
        for (int i = 0; i < cols; i++) {
            if (board[0 * cols + i] == 0) {
                draw = false;
                break;
            }
        }

        if (draw) return state = 0;
        return state = 2;
    }

    void Print(std::ostream& ostr, char RED = 'X', char YELLOW = 'O', char EMPTY = ' ', char VERTICAL_SEP = '|', char HORIZONTAL_SEP = '-', char JOINT = '+') const {
        ostr << VERTICAL_SEP;
        for (int i = 0; i < 7; i++) {
            ostr << i << VERTICAL_SEP;
        }
        ostr << '\n';

        ostr << JOINT;
        for (int i = 0; i < 7; i++) ostr << HORIZONTAL_SEP << JOINT;
        ostr << '\n';

        for (int i = 0; i < rows; i++) {
            ostr << VERTICAL_SEP;
            for (int j = 0; j < cols; j++) {
                if (board[i * cols + j] == 1) ostr << RED;
                else if (board[i * cols + j] == -1) ostr << YELLOW;
                else ostr << EMPTY;

                ostr << VERTICAL_SEP;
            }

            ostr << '\n';
            ostr << JOINT;
            for (int j = 0; j < cols; j++) ostr << HORIZONTAL_SEP << JOINT;
            ostr << '\n';
        }
    }
};
bool operator<(const Connect4& a, const Connect4& b) { return a.GetPosition() < b.GetPosition(); }

class NN_Player {
private:
    int maxd = 3;

    std::map<Connect4, std::pair<double, int>> memo;
public:
    NNet::NeuralNet net;

    NN_Player(const NNet::NeuralNet& net_, int maxd_) : net(net_), maxd(maxd_) {}

    void ResetMemo() {
        memo.clear();
    }
    std::pair<double, int> FindMove(Connect4& game, int d = 0) {
        if (game.State() != 2) {
            return { game.State(), -1 };
        }
        if (d == maxd) {
            std::vector<double> input;

            for (auto& e : game.GetPosition()) input.push_back((double)e);

            return { net.Query(input)(0), -1 };
        }
        //if (memo.find(game) != memo.end()) return memo[game];

        double best = game.Turn() ? INF : -INF;

        auto possible = game.PossibleMoves();
        int best_move = -1;
        for (auto& move : possible) {
            game.Move(move);
            auto found = FindMove(game, d + 1);
            game.UndoMove(move);

            if (game.Turn()) {
                if (found.first < best) {
                    best = found.first;
                    best_move = move;
                }
            }
            else {
                if (found.first > best) {
                    best = found.first;
                    best_move = move;
                }
            }
        }

        return { best, best_move };
    }

    void Learn(const std::vector<std::vector<int>>& positions, int result) {
        for (auto& pos : positions) {
            std::vector<double> in;

            for (auto& e : pos) in.push_back((double)e);
            net.Fit(in, std::vector<double>{(double)result});
        }
    }

    void Save(std::ostream& file) { net.Save(file); }
    void Save(const std::string& path) { net.Save(path); }

    void Load(std::istream& file) { net.Load(file); }
    void Load(const std::string& path) { net.Load(path); }
};

int RandInt(int l, int r) { return rand() % (r - l + 1) + l; }

void HumanVsHuman() {
    Connect4 c4;

    c4.Print(std::cout);
    std::cout << '\n';

    while (true) {
        int move;
        std::cin >> move;

        c4.Move(move);

        c4.Print(std::cout);
        std::cout << '\n';

        if (c4.State() != 2) {
            if (c4.State() == 0) std::cout << "It's a draw!\n";
            else if (c4.State() == 1) std::cout << "Red has won!\n";
            else std::cout << "Yellow has won!\n";

            break;
        }
    }
}
void RandomVsRandom() {
    Connect4 c4;

    srand(time(NULL));

    int epochs;
    std::cin >> epochs;

    int red = 0, yellow = 0, draw = 0;

    while (epochs--) {
        c4.Reset();
        while (true) {
            auto moves = c4.PossibleMoves();
            c4.Move(moves[RandInt(0, moves.size() - 1)]);

            int eval = c4.CheckPosition();
            if (eval != 2) {
                if (eval == -1) yellow++;
                else if (eval == 0) draw++;
                else red++;

                break;
            }
        }
    }

    std::cout << "Red wins: " << red << "\nYellow wins: " << yellow << "\nDraws: " << draw << "\n";
}
void BotVsItself(NN_Player& p1, int epochs, int step) {
    Connect4 game;

    int rwin = 0, ywin = 0, draw = 0;

    //p1.Load(LOCATION + "p12.txt");
    //p2.Load(LOCATION + "p2.txt");

    std::vector<std::vector<int>> poss;
    for (int ep = 1; ep <= epochs; ep++) {
        game.Reset();
        poss.clear();

        while (true) {
            game.Move(p1.FindMove(game).second);
            poss.push_back(game.GetPosition());
            if (game.State() != 2) {
                if (game.State()) rwin++;
                else draw++;

                break;
            }

            game.Move(p1.FindMove(game).second);
            poss.push_back(game.GetPosition());
            if (game.State() != 2) {
                if (game.State()) ywin++;
                else draw++;

                break;
            }
        }


        p1.Learn(poss, game.State());
        p1.ResetMemo();

        if (ep % step == 0) {
            game.Print(std::cout);
            std::cout << '\n';

            std::vector<double> in;
            for (auto& e : game.GetPosition()) in.push_back((double)e);

            std::cout << p1.net.Query(in) << '\n';

            std::cout << "Passed " << ep << " games, stats:\n";
            std::cout << "Red wins: " << rwin << "\nYellow wins: " << ywin << "\nDraws: " << draw << "\n";
        }
    }
}
void RedBotVsYellowPlayer(NN_Player& p1) {
    Connect4 game;
    std::vector<std::vector<int>> poss;

    game.Print(std::cout);
    std::cout << '\n';
    while (true) {
        game.Move(p1.FindMove(game).second);
        poss.push_back(game.GetPosition());

        game.Print(std::cout);
        std::cout << '\n';

        if (game.State() != 2) {
            std::cout << "Bot has won!\n";

            break;
        }

        int move;
        std::cout << "Your move: "; std::cin >> move;

        game.Move(move);
        poss.push_back(game.GetPosition());

        game.Print(std::cout);
        std::cout << '\n';

        if (game.State() != 2) {
            std::cout << "You have won!\n";

            break;
        }
    }

    p1.Learn(poss, game.State());
}
void RedPlayerVsYellowBot(NN_Player& p1) {
    Connect4 game;

    game.Print(std::cout);
    std::cout << '\n';
    while (true) {
        int move;
        std::cout << "Your move: "; std::cin >> move;

        game.Move(move);
        game.Print(std::cout);
        std::cout << '\n';

        if (game.State() != 2) {
            std::cout << "You have won!\n";

            break;
        }

        game.Move(p1.FindMove(game).second);
        game.Print(std::cout);
        std::cout << '\n';

        if (game.State() != 2) {
            std::cout << "Bot has won!\n";

            break;
        }
    }
}

std::string LOCATION = R"(C:\Users\lukad\Desktop\connect4\)";

using namespace NNet;

NeuralNet net{
    6 * 7,
    {
        new DenseL(0.01, 100),
        new ActL(0.02, Tanh, TanhDeriv),
        new DenseL(0.01, 100),
        new ActL(0.02, Tanh, TanhDeriv),
        new DenseL(0.01, 1),
        new ActL(0.02, Tanh, TanhDeriv),
    },
    SqLoss,
    SqLossDeriv
};

NN_Player AI{ net, 3 };

int main()
{   
    //int games, step;
    //std::cin >> games >> step;
    AI.Load(LOCATION + "nugen.txt");
    RedBotVsYellowPlayer(AI);

    //BotVsItself(AI, games, step);

    //AI.Save(LOCATION + "nugen.txt");

    //std::cout << "Red wins: " << rwin << "\nYellow wins: " << ywin << "\nDraws: " << draw << "\n";
}

