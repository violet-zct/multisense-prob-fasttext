/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *               2018-present, Ben Athiwaratkun
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "model.h"

#include <iostream>
#include <assert.h>
#include <algorithm>
#include <random>
#include <cmath>

namespace fasttext {

Model::Model(std::shared_ptr<Matrix> wi,
             std::shared_ptr<Matrix> wo,
             // for variance
             std::shared_ptr<Vector> invar,
             std::shared_ptr<Vector> outvar,
             std::shared_ptr<Args> args,
             int32_t seed)
  : hidden_(args->dim), output_(wo->m_),
  grad_(args->dim), temp_(args->dim), rng(seed), quant_(false)
{
  wi_ = wi;
  wo_ = wo;

  invar_ = invar;
  outvar_ = outvar;

  args_ = args;
  osz_ = wo->m_;
  hsz_ = args->dim;
  negpos = 0;
  loss_ = 0.0;
  nexamples_ = 1;
  initSigmoid();
  initLog();
}

Model::~Model() {
  delete[] t_sigmoid;
  delete[] t_log;
}

void Model::setQuantizePointer(std::shared_ptr<QMatrix> qwi,
                               std::shared_ptr<QMatrix> qwo, bool qout) {
  qwi_ = qwi;
  qwo_ = qwo;
  if (qout) {
    osz_ = qwo_->getM();
  }
}

real Model::binaryLogistic(int32_t target, bool label, real lr) {
  real score = sigmoid(wo_->dotRow(hidden_, target));
  real alpha = lr * (real(label) - score);
  grad_.addRow(*wo_, target, alpha);
  wo_->addRow(hidden_, target, alpha);
  if (label) {
    return -log(score);
  } else {
    return -log(1.0 - score);
  }
}

real Model::negativeSamplingSingleKL(int32_t wordidx, int32_t target, real lr) {
    // Not use this
}

real Model::negativeSamplingSingleVar(int32_t wordidx, int32_t target, real lr) {
  real eplus_result = energy_singleVecvar(wordidx, target, true);
  int32_t negTarget = getNegative(target);
  real eminus_result = energy_singleVecvar(wordidx, negTarget, false);
  real loss = args_->margin - eplus_result + eminus_result;

  if (loss > 0.0){
    real wp_invsumd = 1./(1e-8 + wp_var_sum_);
    real wn_invsumd = 1./(1e-8 + wn_var_sum_);
    real wp_invsumd_square = pow(wp_invsumd, 2.);
    real wn_invsumd_square = pow(wn_invsumd, 2.);

    gradmu_p_.zero();
    gradmu_n_.zero();
    grad_.zero();

    gradvar_p_ = 0.0;
    gradvar_n_ = 0.0;
    gradvar_ = 0.0;

    for (int64_t ii = 0; ii < hidden_.m_; ii++) {
      // eplus_results simplified so only lr left
      gradvar_p_ += 0.5 * lr * (-wp_invsumd + wp_invsumd_square * pow(wp_diff_[ii], 2.));
      gradvar_n_ += -0.5 * lr * (-wn_invsumd + wn_invsumd_square * pow(wn_diff_[ii], 2.));
      gradvar_ += gradvar_p_ + gradvar_n_;
    }

    gradvar_p_ = exp(outvar_->data_[target]) * gradvar_p_;
    gradvar_n_ = exp(outvar_->data_[negTarget] * gradvar_n_);
    gradvar_ = exp(invar_->data_[wordidx]) * gradvar_;
    outvar_->data_[target] += gradvar_p_;
    outvar_->data_[negTarget] += gradvar_n_;
    // TODO: thresholding the variance within a range

    for (int64_t ii = 0; ii < grad_.m_; ii++) {
      gradmu_p_[ii] += lr * (wp_invsumd * wp_diff_[ii]);
      gradmu_n_[ii] += -lr * (wn_invsumd * wn_diff_[ii]);
      grad_.data_[ii] -= (gradmu_p_[ii] + gradmu_n_[ii]);
    }
    if (args_->c != 0.0) {
     gradmu_p_.addRow(wo_, target, -2*lr*args_->c);
     gradmu_n_.addRow(wo_, negTarget, -2*lr*args_->c);
    }

    wo_->addRow(gradmu_p_, target, 1.);
    wo_->addRow(gradmu_n_, negTarget, 1.);

    if (args_->min_logvar !=0 && args_->max_logvar !=0) {
      outvar_->data_[target] = regLogVar(outvar_->data_[target]);
      outvar_->data_[negTarget] = regLogVar(outvar_->data_[negTarget]);
    }
  }
  return std::max((real) 0.0, loss);
}

real Model::regLogVar(real logvar) {
    return std::max(args_->min_logvar, std::min(logvar, args_->max_logvar));
}

real Model::negativeSamplingSingleExpdot(int32_t target, real lr) {
  // loss is the negative of similarity here
  grad_.zero();
  real sim1 = 0.0;
  real sim2 = 0.0;
  real scale = lr/(args_->var_scale);
  int32_t negTarget = getNegative(target);

  sim1 = wo_->dotRow(hidden_, target);
  sim2 = wo_->dotRow(hidden_, negTarget);

  real loss = args_->margin - sim1 + sim2;
  if (loss > 0.0){
    grad_.addRow(*wo_, target, scale);
    grad_.addRow(*wo_, negTarget, -scale);
    
    // Update wo_ itself
    // calculate the loss based on the norm
    wo_->addRow(hidden_, target, scale);
    wo_->addRow(hidden_, negTarget, -scale);
  }
  return std::max((real) 0.0, loss);
}

// partial energy expdot
real Model::partial_energy_vecvar(Vector& hidden, Vector& grad, std::shared_ptr<Matrix> wo, int32_t wordidx, int32_t target, std::shared_ptr<Vector> varin, std::shared_ptr<Vector> varout, bool true_label){
  temp_.zero();
  real var_sum = exp(varin->data_[wordidx]) + exp(varout->data_[target]);

  hidden_.addRow(*wo, target, -1.); // mu - vec
  if true_label == true {
    wp_diff_ = hidden_;
    wp_var_sum_ = var_sum;
  } else {
    wn_diff_ = hidden_;
    wn_var_sum_ = var_sum;
  }
  real sim = 0.0;
  for (int64_t i = 0; i < hidden_.m_; i++) {
    sim += pow(hidden_.data_[i], 2.0)/(1e-8 + var_sum);
  }
  sim += log(var_sum) * args_->dim; // This is the log det part
  sim += args_->dim*log(2*M_PI); // TODO This should be part of the formula
  sim *= -0.5;
  hidden.addRow(*wo, target, 1.); // mu
  return sim;
}

// KL Divergence
real Model::partial_energy_KL(Vector& hidden, Vector& grad, std::shared_ptr<Matrix> wo, int32_t wordidx, int32_t target, std::shared_ptr<Vector> varin, std::shared_ptr<Vector> varout){
  temp_.zero();
  for (int64_t j = 0; j < varin->n_; j++){
    temp_.data_[j] += exp(varin->at(wordidx, j));
  }
  hidden_.addRow(*wo, target, -1.); // mu - vec
  real sim = 0.0;
  for (int64_t i = 0; i < temp_.m_; i++) {
    // TODO
    //sim +=
    sim += pow(hidden_.data_[i], 2.0)/(1e-8 + temp_.data_[i]);
    sim += log(temp_.data_[i]); // This is the log det part
  }
  sim *= -0.5;
  hidden.addRow(*wo, target, 1.); // mu
  return sim;
}

// energy_vecvar but for single case
real Model::energy_singleVecvar(int32_t wordidx, int32_t target, bool true_label) {
  return partial_energy_vecvar(hidden_, grad_, wo_, wordidx, target, invar_, outvar_, true_label);
}

real Model::hierarchicalSoftmax(int32_t target, real lr) {
  real loss = 0.0;
  grad_.zero();
  const std::vector<bool>& binaryCode = codes[target];
  const std::vector<int32_t>& pathToRoot = paths[target];
  for (int32_t i = 0; i < pathToRoot.size(); i++) {
    loss += binaryLogistic(pathToRoot[i], binaryCode[i], lr);
  }
  return loss;
}

void Model::computeOutputSoftmax(Vector& hidden, Vector& output) const {
  if (quant_ && args_->qout) {
    output.mul(*qwo_, hidden);
  } else {
    output.mul(*wo_, hidden);
  }
  real max = output[0], z = 0.0;
  for (int32_t i = 0; i < osz_; i++) {
    max = std::max(output[i], max);
  }
  for (int32_t i = 0; i < osz_; i++) {
    output[i] = exp(output[i] - max);
    z += output[i];
  }
  for (int32_t i = 0; i < osz_; i++) {
    output[i] /= z;
  }
}

void Model::computeOutputSoftmax() {
  computeOutputSoftmax(hidden_, output_);
}

real Model::softmax(int32_t target, real lr) {
  grad_.zero();
  computeOutputSoftmax();
  for (int32_t i = 0; i < osz_; i++) {
    real label = (i == target) ? 1.0 : 0.0;
    real alpha = lr * (label - output_[i]);
    grad_.addRow(*wo_, i, alpha);
    wo_->addRow(hidden_, i, alpha);
  }
  return -log(output_.data_[target]);
}

void Model::computeHidden(const std::vector<int32_t>& input, Vector& hidden)
    const {
  assert(hidden.size() == hsz_);
  hidden.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden.addRow(*wi_, *it);
  }
  hidden.mul(1.0 / input.size());
}

bool Model::comparePairs(const std::pair<real, int32_t> &l,
                         const std::pair<real, int32_t> &r) {
  return l.first > r.first;
}

void Model::predict(const std::vector<int32_t>& input, int32_t k,
                    std::vector<std::pair<real, int32_t>>& heap,
                    Vector& hidden, Vector& output) const {
  assert(k > 0);
  heap.reserve(k + 1);
  computeHidden(input, hidden);
  if (args_->loss == loss_name::hs) {
    dfs(k, 2 * osz_ - 2, 0.0, heap, hidden);
  } else {
    findKBest(k, heap, hidden, output);
  }
  std::sort_heap(heap.begin(), heap.end(), comparePairs);
}

void Model::predict(const std::vector<int32_t>& input, int32_t k,
                    std::vector<std::pair<real, int32_t>>& heap) {
  predict(input, k, heap, hidden_, output_);
}

void Model::findKBest(int32_t k, std::vector<std::pair<real, int32_t>>& heap,
                      Vector& hidden, Vector& output) const {
  computeOutputSoftmax(hidden, output);
  for (int32_t i = 0; i < osz_; i++) {
    if (heap.size() == k && log(output[i]) < heap.front().first) {
      continue;
    }
    heap.push_back(std::make_pair(log(output[i]), i));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
  }
}

void Model::dfs(int32_t k, int32_t node, real score,
                std::vector<std::pair<real, int32_t>>& heap,
                Vector& hidden) const {
  if (heap.size() == k && score < heap.front().first) {
    return;
  }

  if (tree[node].left == -1 && tree[node].right == -1) {
    heap.push_back(std::make_pair(score, node));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
    return;
  }

  real f;
  if (quant_ && args_->qout) {
    f= sigmoid(qwo_->dotRow(hidden, node - osz_));
  } else {
    f= sigmoid(wo_->dotRow(hidden, node - osz_));
  }

  dfs(k, tree[node].left, score + log(1.0 - f), heap, hidden);
  dfs(k, tree[node].right, score + log(f), heap, hidden);
}

float probRand() {
    static thread_local std::mt19937 generator;
    std::uniform_int_distribution<int> distribution(0,1000);
    return distribution(generator)/(1.*1000);
}

void Model::update(const std::vector<int32_t>& input, int32_t target, real lr) {
  assert(target >= 0);
  assert(target < osz_);
  if (input.size() == 0) return;

  // get the word index --> this is the first element in 'input'
  int32_t wordidx = 0;
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wordidx = *it;
    break;
  }
  
  computeHidden(input, hidden_);
  if (args_->loss == loss_name::ns) {
    loss_ += negativeSamplingSingleVar(wordidx, target, lr);
  } else if (args_->loss == loss_name::hs) {
    // not using
    loss_ += hierarchicalSoftmax(target, lr);
  } else {
    // not using
    loss_ += softmax(target, lr);
  }
  nexamples_ += 1;

  if (args_->norm_grad) {
    grad_.mul(1.0 / input.size());
  }

  if (args_->c != 0.0) {
    grad_.addRow(wi_, wordidx, -2*lr*args_->c);
  }

  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi_->addRow(grad_, *it, 1.0);
  }

  invar_->data_[wordidx] += gradvar_;

  if (args_->min_logvar !=0 && args_->max_logvar !=0) {
    invar_->data_[wordidx] = regLogVar(invar_->data_[wordidx]);
  }
}

void Model::groupSparsityRegularization(int min, int max, int num_gs_samples, double strength){
  // sampling from the uniform interval [min, max)
  grad_.zero();
  std::uniform_int_distribution<> uniform(min, max-1);
  if (args_->gs_lambda > 1e-12){
    for (int ii = 0; ii < num_gs_samples; ii++){
      // Note: osz_ is the number of words in the dictionary
      // Perhaps adjusts the distribution of this sampler
      int32_t idx = uniform(rng);
      loss_ += groupSparsityRegularization(strength, idx);
    }
  }
}

real Model::groupSparsityRegularization(double reg_strength, int32_t word){
  // To be efficient, only do it if the strength is non-zero
  if (reg_strength > 0.0000000001) {
    real norm = wi_->l2NormRow(word);
    real loss = reg_strength*norm;
    // 2. update the wi_ accordingly based on the gradient
    // note: reuse the grad variable here
    grad_.zero();
    grad_.addRow(*wi_, word, -reg_strength/(norm + 0.00001));
    wi_->addRow(grad_, word, 1.0);
    return loss;
  } else {
    return 0.0;
  }
}

void Model::setTargetCounts(const std::vector<int64_t>& counts) {
  assert(counts.size() == osz_);
  if (args_->loss == loss_name::ns) {
    initTableNegatives(counts);
  }
  if (args_->loss == loss_name::hs) {
    buildTree(counts);
  }
}

void Model::initTableNegatives(const std::vector<int64_t>& counts) {
  real z = 0.0;
  for (size_t i = 0; i < counts.size(); i++) {
    z += pow(counts[i], 0.5);
  }
  for (size_t i = 0; i < counts.size(); i++) {
    real c = pow(counts[i], 0.5);
    for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
      negatives.push_back(i);
    }
  }
  std::shuffle(negatives.begin(), negatives.end(), rng);
}

int32_t Model::getNegative(int32_t target) {
  int32_t negative;
  do {
    negative = negatives[negpos];
    negpos = (negpos + 1) % negatives.size();
  } while (target == negative);
  return negative;
}

void Model::buildTree(const std::vector<int64_t>& counts) {
  tree.resize(2 * osz_ - 1);
  for (int32_t i = 0; i < 2 * osz_ - 1; i++) {
    tree[i].parent = -1;
    tree[i].left = -1;
    tree[i].right = -1;
    tree[i].count = 1e15;
    tree[i].binary = false;
  }
  for (int32_t i = 0; i < osz_; i++) {
    tree[i].count = counts[i];
  }
  int32_t leaf = osz_ - 1;
  int32_t node = osz_;
  for (int32_t i = osz_; i < 2 * osz_ - 1; i++) {
    int32_t mini[2];
    for (int32_t j = 0; j < 2; j++) {
      if (leaf >= 0 && tree[leaf].count < tree[node].count) {
        mini[j] = leaf--;
      } else {
        mini[j] = node++;
      }
    }
    tree[i].left = mini[0];
    tree[i].right = mini[1];
    tree[i].count = tree[mini[0]].count + tree[mini[1]].count;
    tree[mini[0]].parent = i;
    tree[mini[1]].parent = i;
    tree[mini[1]].binary = true;
  }
  for (int32_t i = 0; i < osz_; i++) {
    std::vector<int32_t> path;
    std::vector<bool> code;
    int32_t j = i;
    while (tree[j].parent != -1) {
      path.push_back(tree[j].parent - osz_);
      code.push_back(tree[j].binary);
      j = tree[j].parent;
    }
    paths.push_back(path);
    codes.push_back(code);
  }
}

real Model::getLoss() const {
  return loss_ / nexamples_;
}

void Model::initSigmoid() {
  t_sigmoid = new real[SIGMOID_TABLE_SIZE + 1];
  for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
    real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
    t_sigmoid[i] = 1.0 / (1.0 + std::exp(-x));
  }
}

void Model::initLog() {
  t_log = new real[LOG_TABLE_SIZE + 1];
  for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
    real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
    t_log[i] = std::log(x);
  }
}

real Model::log(real x) const {
  if (x > 1.0) {
    return 0.0;
  }
  int i = int(x * LOG_TABLE_SIZE);
  return t_log[i];
}

real Model::sigmoid(real x) const {
  if (x < -MAX_SIGMOID) {
    return 0.0;
  } else if (x > MAX_SIGMOID) {
    return 1.0;
  } else {
    int i = int((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
    return t_sigmoid[i];
  }
}

void Model::expVar() {
  int64_t m = invar_->m_;
  assert(m == outvar_->m_);
  for (int32_t i = 0; i < m; ++i) {
    invar_->data_[i] = exp(invar_->data_[i]);
    outvar_->data_[i] = exp(outvar_->data_[i]);
  }
}
}
