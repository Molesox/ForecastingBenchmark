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
    std::vector<char> m_chars;

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

        // std::cout << "v = (" << v.n_rows << "," << v.n_cols << ")" << std::endl;
        // std::cout << "maxord = " << maxord <<std::endl;

        for (size_t ord = 0; ord <= maxord; ord++)
        {
            if (v.n_rows > ord)
            {
                if (std::find(m_chars.begin(), m_chars.end(), v.back()) == m_chars.end()) //just to tell elem not in vect...
                {
                    m_chars.push_back(v.back());
                }
                // std::cout << v.rows(v.n_rows - ord -1, v.n_rows -1).t() <<std::endl;
                std::vector<size_t> seq = arma::conv_to<std::vector<size_t>>::from(v.rows(v.n_rows - ord - 1, v.n_rows - 1));
                // std::cout<< "------" <<std::endl;

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
        arma::vec scores(nb_chars, arma::fill::zeros);
        arma::vec context;

        for(size_t ord = maxord; ord > 0; ord --)
        {
            scores.zeros();
            context = v.rows(v.n_rows - ord + 1, v.n_rows - 1);

            for (size_t i = 0; i < nb_chars; i++)
            {
                context.insert_rows(context.n_rows,m_chars[i]);
            }
            
        }

    //   nchar = length(m.chars);
    // for ord=m.maxord:-1:0
    //     %     fprintf('for order %d\n', ord);
    //     scores = zeros(1, nchar);
    //     context = v(length(v)-ord+1:end);
    //     for i=1:nchar
    //         seq = num2str([context m.chars(i)]);
    //         if isKey(m.maps{ord+1},seq)
    //             scores(i) = m.maps{ord+1}(seq);
    //         end
    //     end
    //     %     fprintf('scores: %s\n', num2str(scores));
    //     if ~all(scores == 0)
    //         [~, best_i] = max(scores);
    //         best_char = m.chars(best_i);
    //         break
    //     end
    //     %     fprintf('ALL ZERO: DROPPING TO LEVEL %d\n', ord-1);
    // end

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