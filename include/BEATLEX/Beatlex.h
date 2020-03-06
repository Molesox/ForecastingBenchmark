#ifndef BEATLEX_BEATLEX_H
#define BEATLEX_BEATLEX_H

#include <armadillo>
#include <algorithm>
#include <tuple>
#include <vector>
#include <list>
#include <map>
#include <iostream>
#include <iomanip>

#define VERBOSY false

void loggy(std::string text);
struct dtw_t
{
    arma::mat D;  //accumulated distance matrix
    arma::umat w; //optimal path
    double k;     //normalizing factor
    double Dist;  //unnormalized distance between t and r

    void print()
    {
        std::cout << "dtw:: D = (" << D.n_rows << "," << D.n_cols << ")" << std::endl;
        // std::cout.precision(1);
        // std::cout.setf(std::ios::fixed);
        // D.col(0).raw_print(std::cout);
        std::cout << "dtw:: w = (" << w.n_rows << "," << w.n_cols << ")" << std::endl;
        std::cout << "dtw:: k = " << std::setprecision(6) << k << std::endl;
        std::cout << "dtw:: Dist = " << std::setprecision(7) << Dist << std::endl;
    }
};

dtw_t dtw(arma::mat &t, arma::mat &r, size_t max_dis);
arma::mat distance(const arma::mat &X, const arma::mat &Y);

typedef std::vector<std::map<std::vector<size_t>, int>> vmaps_t;
struct markov
{
    uint8_t maxord;
    vmaps_t m_maps;
    std::vector<size_t> m_chars;

    void print_map()
    {
        for (const auto map : m_maps)
        {
            for (auto it = map.begin(); it != map.end(); it++)
            {
                auto key = it->first;
                auto value = it->second;
                int curr = 1;
                std::cout << "(";
                for (auto el : key)
                {
                    std::cout << el;
                    if (curr < key.size())
                        std::cout << ", ";
                    curr++;
                }
                std::cout << ")";
                std::cout << " val = " << value << std::endl;
            }
        }
    }
    markov(uint8_t maxord) : maxord(maxord)
    {
        m_maps = vmaps_t(maxord + 1);
    }
    void update(const arma::vec &v)
    {

        for (size_t ord = 0; ord <= maxord; ord++)
        {
            if (v.n_rows > ord)
            {
                if (std::find(m_chars.begin(), m_chars.end(), v.back()) == m_chars.end()) //elem not in vect.
                {
                    m_chars.push_back(v.back());
                }
                std::vector<size_t> seq = arma::conv_to<std::vector<size_t>>::from(v.rows(v.n_rows - ord - 1, v.n_rows - 1));
                if (m_maps[ord].find(seq) == m_maps[ord].end())
                {
                    m_maps[ord][seq] = 0;
                }
                m_maps[ord][seq] = m_maps[ord][seq] + 1;
            }
        }
    }
    static std::string vec2string(const arma::vec &v)
    {
        std::string ret;
        for (const auto &el : v)
        {
            ret.append(std::to_string(el));
        }
        return ret;
    }
    size_t predict(const arma::vec &v)
    {
        size_t nb_chars = m_chars.size();
        size_t best_char = 0;
        arma::uvec scores(nb_chars, arma::fill::zeros);
        arma::vec context;

        for (size_t ord = maxord; ord > 0; ord--)
        {
            scores.zeros();
            context = v.rows(v.n_rows - ord, v.n_rows - 1);
            context.resize(context.size() + 1);
            std::vector<size_t> seq;
            for (size_t i = 0; i < nb_chars; i++)
            {
                context(context.n_rows - 1) = m_chars[i];
                seq = arma::conv_to<std::vector<size_t>>::from(context);

                if (m_maps[ord].find(seq) != m_maps[ord].end())
                {
                    scores(i) = m_maps[ord][seq];
                }
            }
            if (not arma::all(scores))//There is at least 1 non-zero value.
            {
                best_char = m_chars[scores.index_max()];
                break;
            }
        }
        return best_char;
    }
};

class Beatlex
{
public:
    const arma::mat &X;
    arma::mat Xp;
    std::vector<arma::mat> models;

    std::vector<size_t> starts;
    std::vector<size_t> ends;
    std::vector<size_t> idx;

    size_t Smin;
    size_t Smax;
    size_t maxdist;
    size_t pred_steps;

    double totalerr;
    double model_momentum;
    double best_prefix_length;
    size_t max_vocab;

public:
    Beatlex(arma::mat &data, size_t smin, size_t smax, size_t maxdist, size_t predsteps);
    std::tuple<size_t, size_t> new_segment(size_t cur);
    void summarize_seq();
    arma::mat forecast();
};

#endif //BEATLEX_BEATLEX_H