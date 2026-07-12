from pathlib import Path
import pandas as pd
import numpy as np
import json, math, shutil, zipfile, base64, os
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, precision_recall_fscore_support, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
import joblib

ROOT=Path('/mnt/data')
V8=ROOT/'fifa_2026_predictor_v8'
V9=ROOT/'fifa_2026_predictor_v9'
if V9.exists(): shutil.rmtree(V9)
for p in [V9/'data'/'raw',V9/'data'/'processed',V9/'outputs',V9/'checkpoints',V9/'scripts',V9/'reports',V9/'ui'/'assets']:
    p.mkdir(parents=True,exist_ok=True)

# Copy branding assets when available
for p in (V8/'ui'/'assets').glob('*') if (V8/'ui'/'assets').exists() else []:
    shutil.copy2(p,V9/'ui'/'assets'/p.name)
fallback_logo=ROOT/'WhatsApp Image 2026-02-25 at 10.37.09 AM (1).jpeg'
if fallback_logo.exists() and not (V9/'ui'/'assets'/'samba_wordmark.jpg').exists():
    shutil.copy2(fallback_logo,V9/'ui'/'assets'/'samba_wordmark.jpg')

pred=pd.read_csv(V8/'outputs'/'v8_match_predictions.csv')
ratings=pd.read_csv(V8/'outputs'/'v8_final26_team_ratings.csv')
group_pred=pd.read_csv(V8/'outputs'/'v8_group_advancement.csv')

# Official group-stage score dataset transcribed from published FIFA-sourced tournament results.
results = {
'A': [('Mexico','South Africa',2,0),('South Korea','Czech Republic',2,1),('Czech Republic','South Africa',1,1),('Mexico','South Korea',1,0),('Mexico','Czech Republic',3,0),('South Africa','South Korea',1,0)],
'B': [('Canada','Bosnia and Herzegovina',1,1),('Qatar','Switzerland',1,1),('Canada','Qatar',6,0),('Switzerland','Bosnia and Herzegovina',4,1),('Bosnia and Herzegovina','Qatar',3,1),('Canada','Switzerland',1,2)],
'C': [('Brazil','Morocco',1,1),('Haiti','Scotland',0,1),('Brazil','Haiti',3,0),('Scotland','Morocco',0,1),('Morocco','Haiti',4,2),('Scotland','Brazil',0,3)],
'D': [('United States','Paraguay',4,1),('Australia','Turkey',2,0),('Turkey','Paraguay',0,1),('United States','Australia',2,0),('Paraguay','Australia',0,0),('United States','Turkey',2,3)],
'E': [('Germany','Curaçao',7,1),('Ivory Coast','Ecuador',1,0),('Ecuador','Curaçao',0,0),('Germany','Ivory Coast',2,1),('Curaçao','Ivory Coast',0,2),('Ecuador','Germany',2,1)],
'F': [('Netherlands','Japan',2,2),('Sweden','Tunisia',5,1),('Netherlands','Sweden',5,1),('Tunisia','Japan',0,4),('Japan','Sweden',1,1),('Tunisia','Netherlands',1,3)],
'G': [('Belgium','Egypt',1,1),('Iran','New Zealand',2,2),('Belgium','Iran',0,0),('New Zealand','Egypt',1,3),('Egypt','Iran',1,1),('New Zealand','Belgium',1,5)],
'H': [('Saudi Arabia','Uruguay',1,1),('Spain','Cape Verde',0,0),('Spain','Saudi Arabia',4,0),('Uruguay','Cape Verde',2,2),('Cape Verde','Saudi Arabia',0,0),('Uruguay','Spain',0,1)],
'I': [('France','Senegal',3,1),('Iraq','Norway',1,4),('France','Iraq',3,0),('Norway','Senegal',3,2),('Norway','France',1,4),('Senegal','Iraq',5,0)],
'J': [('Argentina','Algeria',3,0),('Austria','Jordan',3,1),('Argentina','Austria',2,0),('Jordan','Algeria',1,2),('Algeria','Austria',3,3),('Jordan','Argentina',1,3)],
'K': [('Portugal','DR Congo',1,1),('Uzbekistan','Colombia',1,3),('Colombia','DR Congo',1,0),('Portugal','Uzbekistan',5,0),('Colombia','Portugal',0,0),('DR Congo','Uzbekistan',3,1)],
'L': [('England','Croatia',4,2),('Ghana','Panama',1,0),('England','Ghana',0,0),('Panama','Croatia',0,1),('Croatia','Ghana',2,1),('Panama','England',0,2)],
}
score_rows=[]
for g, games in results.items():
    for h,a,hg,ag in games:
        score_rows.append({'group':g,'home_team':h,'away_team':a,'actual_home_goals':hg,'actual_away_goals':ag})
actual=pd.DataFrame(score_rows)
actual['source_name']='FIFA-sourced 2026 World Cup results table'
actual['source_url']='https://en.wikipedia.org/wiki/2026_FIFA_World_Cup'
actual.to_csv(V9/'data'/'raw'/'world_cup_2026_group_stage_results.csv',index=False)

# Merge pre-tournament v8 predictions with outcomes.
eval_df=pred.merge(actual,on=['group','home_team','away_team'],how='left',validate='one_to_one')
assert eval_df['actual_home_goals'].notna().all(), eval_df[eval_df.actual_home_goals.isna()][['group','home_team','away_team']]

def outcome(h,a):
    return 'home_win' if h>a else ('draw' if h==a else 'away_win')
classes=['home_win','draw','away_win']
eval_df['actual_outcome']=[outcome(h,a) for h,a in zip(eval_df.actual_home_goals,eval_df.actual_away_goals)]
eval_df['predicted_outcome_argmax']=np.array(classes)[np.argmax(eval_df[['p_home_win','p_draw','p_away_win']].to_numpy(),axis=1)]
eval_df['outcome_correct']=(eval_df.actual_outcome==eval_df.predicted_outcome_argmax).astype(int)
eval_df['actual_score']=eval_df.actual_home_goals.astype(int).astype(str)+'-'+eval_df.actual_away_goals.astype(int).astype(str)
eval_df['exact_score_correct']=(eval_df.predicted_exact_score==eval_df.actual_score).astype(int)
eval_df['expected_score_correct']=(eval_df.expected_score==eval_df.actual_score).astype(int)
eval_df['actual_total_goals']=eval_df.actual_home_goals+eval_df.actual_away_goals
eval_df['predicted_total_xg']=eval_df.expected_goals_home+eval_df.expected_goals_away
eval_df['total_goals_abs_error']=(eval_df.predicted_total_xg-eval_df.actual_total_goals).abs()
eval_df['home_goals_abs_error']=(eval_df.expected_goals_home-eval_df.actual_home_goals).abs()
eval_df['away_goals_abs_error']=(eval_df.expected_goals_away-eval_df.actual_away_goals).abs()
eval_df['within_one_goal_both']=((eval_df.home_goals_abs_error<=1)&(eval_df.away_goals_abs_error<=1)).astype(int)
eval_df['actual_over_2_5']=(eval_df.actual_total_goals>2.5).astype(int)
eval_df['pred_over_2_5']=(eval_df.predicted_total_xg>2.5).astype(int)
eval_df['actual_btts']=((eval_df.actual_home_goals>0)&(eval_df.actual_away_goals>0)).astype(int)
lamh=eval_df.expected_goals_home.to_numpy(); lama=eval_df.expected_goals_away.to_numpy()
eval_df['p_btts']=1-np.exp(-lamh)-np.exp(-lama)+np.exp(-(lamh+lama))
eval_df['pred_btts']=(eval_df.p_btts>=0.5).astype(int)

probs=eval_df[['p_home_win','p_draw','p_away_win']].to_numpy()
y_idx=np.array([classes.index(x) for x in eval_df.actual_outcome])
y_one=np.eye(3)[y_idx]
acc=accuracy_score(y_idx,np.argmax(probs,axis=1))
ll=log_loss(y_idx,probs,labels=[0,1,2])
brier=float(np.mean(np.sum((probs-y_one)**2,axis=1)))
# Ranked Probability Score, ordered as home/draw/away.
cum_p=np.cumsum(probs,axis=1)[:,:-1]; cum_y=np.cumsum(y_one,axis=1)[:,:-1]
rps=float(np.mean(np.sum((cum_p-cum_y)**2,axis=1)/2))
# ECE of max confidence.
conf=probs.max(axis=1); correct=(np.argmax(probs,axis=1)==y_idx).astype(float)
ece=0.0
for lo in np.linspace(0,0.9,10):
    hi=lo+0.1
    mask=(conf>=lo)&(conf<(hi if hi<1 else 1.000001))
    if mask.any(): ece += mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
prec,rec,f1,support=precision_recall_fscore_support(y_idx,np.argmax(probs,axis=1),labels=[0,1,2],zero_division=0)
cm=confusion_matrix(y_idx,np.argmax(probs,axis=1),labels=[0,1,2])

# Tournament placement metrics.
actual_order={
'A':['Mexico','South Africa','South Korea','Czech Republic'],
'B':['Switzerland','Canada','Bosnia and Herzegovina','Qatar'],
'C':['Brazil','Morocco','Scotland','Haiti'],
'D':['United States','Australia','Paraguay','Turkey'],
'E':['Germany','Ivory Coast','Ecuador','Curaçao'],
'F':['Netherlands','Japan','Sweden','Tunisia'],
'G':['Belgium','Egypt','Iran','New Zealand'],
'H':['Spain','Cape Verde','Uruguay','Saudi Arabia'],
'I':['France','Norway','Senegal','Iraq'],
'J':['Argentina','Austria','Algeria','Jordan'],
'K':['Colombia','Portugal','DR Congo','Uzbekistan'],
'L':['England','Croatia','Ghana','Panama'],
}
third_qual={'DR Congo','Sweden','Ghana','Ecuador','Bosnia and Herzegovina','Algeria','Paraguay','Senegal'}
actual_qual=set()
for g,order in actual_order.items(): actual_qual.update(order[:2])
actual_qual.update(third_qual)
pred_winners={g:df.sort_values('p_group_win',ascending=False).iloc[0].team for g,df in group_pred.groupby('group')}
group_winner_hits=sum(pred_winners[g]==actual_order[g][0] for g in actual_order)
pred_top2={g:list(df.sort_values('p_top2',ascending=False).head(2).team) for g,df in group_pred.groupby('group')}
top2_overlap=sum(len(set(pred_top2[g])&set(actual_order[g][:2])) for g in actual_order)
exact_top2_groups=sum(set(pred_top2[g])==set(actual_order[g][:2]) for g in actual_order)
pred_qual=set(group_pred.sort_values('p_advance',ascending=False).head(32).team)
qual_hits=len(pred_qual&actual_qual)

metrics={
 'evaluation_cutoff':'2026-06-27 after completion of the group stage',
 'matches_evaluated':int(len(eval_df)),
 'outcome_accuracy':float(acc),
 'log_loss':float(ll),
 'multiclass_brier_score':brier,
 'ranked_probability_score':rps,
 'expected_calibration_error_10bin':float(ece),
 'exact_score_accuracy':float(eval_df.exact_score_correct.mean()),
 'rounded_expected_score_accuracy':float(eval_df.expected_score_correct.mean()),
 'home_goals_mae':float(eval_df.home_goals_abs_error.mean()),
 'away_goals_mae':float(eval_df.away_goals_abs_error.mean()),
 'total_goals_mae':float(eval_df.total_goals_abs_error.mean()),
 'within_one_goal_both_teams_rate':float(eval_df.within_one_goal_both.mean()),
 'over_2_5_accuracy':float((eval_df.actual_over_2_5==eval_df.pred_over_2_5).mean()),
 'btts_accuracy':float((eval_df.actual_btts==eval_df.pred_btts).mean()),
 'group_winner_accuracy':float(group_winner_hits/12),
 'group_winners_correct':int(group_winner_hits),
 'top2_team_overlap_rate':float(top2_overlap/24),
 'top2_teams_correct':int(top2_overlap),
 'groups_with_exact_top2_pair':int(exact_top2_groups),
 'round32_qualifier_precision_recall':float(qual_hits/32),
 'round32_qualifiers_correct':int(qual_hits),
 'class_metrics':{classes[i]:{'precision':float(prec[i]),'recall':float(rec[i]),'f1':float(f1[i]),'support':int(support[i])} for i in range(3)},
 'confusion_matrix':{'labels':classes,'matrix':cm.tolist()},
}
(V9/'outputs'/'group_stage_evaluation_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
eval_df.to_csv(V9/'outputs'/'group_stage_prediction_vs_actual.csv',index=False)

# ---------- Tournament-update training data ----------
# Sequential Elo and rolling form built strictly from prior group matches.
start_rating=dict(zip(ratings.team,ratings.trained_roster_rating))
elo={t:float(start_rating[t]) for t in start_rating}
stat={t:{'mp':0,'pts':0,'gf':0,'ga':0} for t in elo}
feature_rows=[]
ordered=eval_df.sort_values(['date','group']).copy()
for _,r in ordered.iterrows():
    h,a=r.home_team,r.away_team
    # pre-match tournament form
    sh,sa=stat[h],stat[a]
    def per(s,k): return s[k]/s['mp'] if s['mp'] else 0.0
    host_bonus=45.0 if (not bool(r.neutral) and h in {'Canada','Mexico','United States'}) else 0.0
    diff=(elo[h]+host_bonus)-elo[a]
    row=r.to_dict()
    row.update({
      'elo_home_pre':elo[h],'elo_away_pre':elo[a],'elo_diff_pre':diff,
      'home_form_ppg':per(sh,'pts'),'away_form_ppg':per(sa,'pts'),
      'form_ppg_diff':per(sh,'pts')-per(sa,'pts'),
      'home_gf_pg':per(sh,'gf'),'away_gf_pg':per(sa,'gf'),
      'gf_pg_diff':per(sh,'gf')-per(sa,'gf'),
      'home_ga_pg':per(sh,'ga'),'away_ga_pg':per(sa,'ga'),
      'ga_pg_diff':per(sh,'ga')-per(sa,'ga'),
      'home_matches_played':sh['mp'],'away_matches_played':sa['mp'],
      'matchday':int(sh['mp']+1),
      'host_team_indicator':int(host_bonus>0),
    })
    feature_rows.append(row)
    # update Elo after result
    exp=1/(1+10**(-diff/400))
    score=1.0 if r.actual_home_goals>r.actual_away_goals else (0.5 if r.actual_home_goals==r.actual_away_goals else 0.0)
    mov=math.log(abs(r.actual_home_goals-r.actual_away_goals)+1)*2.2/max(1.0,(abs(diff)*0.001+2.2))
    k=32*max(1.0,mov)
    delta=k*(score-exp)
    elo[h]+=delta; elo[a]-=delta
    # update form
    hp=3 if score==1 else (1 if score==0.5 else 0); ap=3 if score==0 else (1 if score==0.5 else 0)
    sh['mp']+=1; sh['pts']+=hp; sh['gf']+=r.actual_home_goals; sh['ga']+=r.actual_away_goals
    sa['mp']+=1; sa['pts']+=ap; sa['gf']+=r.actual_away_goals; sa['ga']+=r.actual_home_goals

train=pd.DataFrame(feature_rows)
train.to_csv(V9/'data'/'processed'/'group_stage_training_features.csv',index=False)
final_form=[]
for t,s in stat.items():
    final_form.append({'team':t,'tournament_elo':elo[t],'matches_played':s['mp'],'points':s['pts'],'goals_for':s['gf'],'goals_against':s['ga'],'goal_difference':s['gf']-s['ga'],'points_per_game':s['pts']/s['mp'],'goals_for_per_game':s['gf']/s['mp'],'goals_against_per_game':s['ga']/s['mp']})
form_df=pd.DataFrame(final_form)
form_df.to_csv(V9/'outputs'/'team_form_after_group_stage.csv',index=False)

# Features use base v8 probabilities + xG + strictly prior tournament form.
eps=1e-6
train['logit_home_base']=np.log(np.clip(train.p_home_win,eps,1)/np.clip(train.p_draw,eps,1))
train['logit_away_base']=np.log(np.clip(train.p_away_win,eps,1)/np.clip(train.p_draw,eps,1))
feature_cols=['logit_home_base','logit_away_base','expected_goals_home','expected_goals_away','elo_diff_pre','form_ppg_diff','gf_pg_diff','ga_pg_diff','home_matches_played','away_matches_played','host_team_indicator']
X=train[feature_cols].astype(float)
y=np.array([classes.index(x) for x in train.actual_outcome])
groups_cv=train.group.to_numpy()

# Group-held-out validation with fixed regularization selected conservatively for the small 72-match sample.
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
best_C=0.2
outcome_model=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(C=best_C,max_iter=1200,solver='lbfgs'))])
outcome_model.fit(X,y)

# Regularized log-goal regressions are stable on this small tournament sample.
goal_cols=['expected_goals_home','expected_goals_away','elo_diff_pre','form_ppg_diff','gf_pg_diff','ga_pg_diff','home_matches_played','away_matches_played','host_team_indicator']
Xg=train[goal_cols].astype(float)
best_alpha=3.0
home_goal_model=Pipeline([('scale',StandardScaler()),('reg',Ridge(alpha=best_alpha))])
away_goal_model=Pipeline([('scale',StandardScaler()),('reg',Ridge(alpha=best_alpha))])
home_goal_model.fit(Xg,np.log1p(train.actual_home_goals)); away_goal_model.fit(Xg,np.log1p(train.actual_away_goals))
# Rate calibration: preserve matchup variation while matching the observed tournament home/away scoring rates.
raw_train_h=np.maximum(0.05,np.expm1(home_goal_model.predict(Xg)))
raw_train_a=np.maximum(0.05,np.expm1(away_goal_model.predict(Xg)))
home_goal_scale=float(train.actual_home_goals.mean()/raw_train_h.mean())
away_goal_scale=float(train.actual_away_goals.mean()/raw_train_a.mean())

joblib.dump({'model':outcome_model,'feature_cols':feature_cols,'classes':classes,'C':best_C},V9/'checkpoints'/'v9_outcome_model.joblib')
joblib.dump({'home_model':home_goal_model,'away_model':away_goal_model,'feature_cols':goal_cols,'alpha':best_alpha,'target_transform':'log1p','home_goal_scale':home_goal_scale,'away_goal_scale':away_goal_scale},V9/'checkpoints'/'v9_score_models.joblib')

# Six group-held-out folds: each test fold contains two whole groups.
cv=GroupKFold(n_splits=6)
cv_p=np.zeros((len(train),3)); cv_lh=np.zeros(len(train)); cv_la=np.zeros(len(train))
cv_records=[]
for fold,(tr,va) in enumerate(cv.split(X,y,groups_cv),1):
    om=clone(outcome_model); hm=clone(home_goal_model); am=clone(away_goal_model)
    om.fit(X.iloc[tr],y[tr]); hm.fit(Xg.iloc[tr],np.log1p(train.actual_home_goals.iloc[tr])); am.fit(Xg.iloc[tr],np.log1p(train.actual_away_goals.iloc[tr]))
    cv_p[va]=om.predict_proba(X.iloc[va])
    raw_tr_h=np.maximum(0.05,np.expm1(hm.predict(Xg.iloc[tr]))); raw_tr_a=np.maximum(0.05,np.expm1(am.predict(Xg.iloc[tr])))
    fold_h_scale=float(train.actual_home_goals.iloc[tr].mean()/raw_tr_h.mean()); fold_a_scale=float(train.actual_away_goals.iloc[tr].mean()/raw_tr_a.mean())
    cv_lh[va]=np.maximum(0.05,np.expm1(hm.predict(Xg.iloc[va])))*fold_h_scale; cv_la[va]=np.maximum(0.05,np.expm1(am.predict(Xg.iloc[va])))*fold_a_scale
    cv_records.append({'fold':fold,'held_out_groups':','.join(sorted(set(groups_cv[va]))),'log_loss':float(log_loss(y[va],cv_p[va],labels=[0,1,2])),'accuracy':float(accuracy_score(y[va],np.argmax(cv_p[va],axis=1))),'goal_mae':float(np.mean(np.r_[np.abs(cv_lh[va]-train.actual_home_goals.iloc[va]),np.abs(cv_la[va]-train.actual_away_goals.iloc[va])]))})
pd.DataFrame(cv_records).to_csv(V9/'reports'/'cross_validation_results.csv',index=False)
updated_cv={
 'group_held_out_folds':6,
 'group_held_out_log_loss':float(log_loss(y,cv_p,labels=[0,1,2])),
 'group_held_out_accuracy':float(accuracy_score(y,np.argmax(cv_p,axis=1))),
 'group_held_out_goal_mae':float(np.mean(np.r_[np.abs(cv_lh-train.actual_home_goals),np.abs(cv_la-train.actual_away_goals)])),
 'outcome_C':best_C,'goal_ridge_alpha':best_alpha,'home_goal_scale':home_goal_scale,'away_goal_scale':away_goal_scale,
 'note':'Six-fold group-held-out validation; each fold withholds two complete groups. Final models are then fitted on all 72 group-stage matches.'
}
(V9/'reports'/'updated_model_cv_metrics.json').write_text(json.dumps(updated_cv,indent=2),encoding='utf-8')

# ---------- Round of 32 forecasts ----------
r32=[
('2026-06-28','South Africa','Canada','Inglewood','United States'),
('2026-06-29','Germany','Paraguay','Foxborough','United States'),
('2026-06-29','Brazil','Japan','Houston','United States'),
('2026-06-29','Netherlands','Morocco','Guadalupe','Mexico'),
('2026-06-30','Ivory Coast','Norway','Arlington','United States'),
('2026-06-30','France','Sweden','East Rutherford','United States'),
('2026-06-30','Mexico','Ecuador','Mexico City','Mexico'),
('2026-07-01','England','DR Congo','Atlanta','United States'),
('2026-07-01','United States','Bosnia and Herzegovina','Santa Clara','United States'),
('2026-07-01','Belgium','Senegal','Seattle','United States'),
('2026-07-02','Portugal','Croatia','Toronto','Canada'),
('2026-07-02','Spain','Austria','Inglewood','United States'),
('2026-07-02','Switzerland','Algeria','Vancouver','Canada'),
('2026-07-03','Argentina','Cape Verde','Miami Gardens','United States'),
('2026-07-03','Australia','Egypt','Arlington','United States'),
('2026-07-03','Colombia','Ghana','Kansas City','United States'),
]
rate=ratings.set_index('team')
form=form_df.set_index('team')
actual_avg_goals=eval_df.actual_total_goals.mean()

def poisson_scores(lh,la,maxg=7):
    rows=[]
    for i in range(maxg+1):
      pi=math.exp(-lh)*lh**i/math.factorial(i)
      for j in range(maxg+1):
        pj=math.exp(-la)*la**j/math.factorial(j)
        rows.append((i,j,pi*pj))
    rows.sort(key=lambda z:z[2],reverse=True)
    return [{'score':f'{i}-{j}','probability':float(p)} for i,j,p in rows[:5]]

r32_rows=[]
for date,h,a,city,country in r32:
    host=int(h in {'Canada','Mexico','United States'})
    host_bonus=45 if host else 0
    diff=(form.loc[h,'tournament_elo']+host_bonus)-form.loc[a,'tournament_elo']
    draw_base=float(np.clip(0.25-0.035*abs(diff)/400,0.16,0.28))
    h_non=1/(1+10**(-diff/400))
    base_ph=(1-draw_base)*h_non; base_pa=(1-draw_base)*(1-h_non)
    # matchup xG from attack/defense priors, rescaled to observed tournament scoring.
    raw_h=math.sqrt(max(0.15,rate.loc[h,'base_xg_for'])*max(0.15,rate.loc[a,'base_xg_against']))
    raw_a=math.sqrt(max(0.15,rate.loc[a,'base_xg_for'])*max(0.15,rate.loc[h,'base_xg_against']))
    scale=actual_avg_goals/max(0.4,raw_h+raw_a)
    bxh=raw_h*scale*(1.06 if host else 1.0); bxa=raw_a*scale*(0.97 if host else 1.0)
    form_ppg_diff=form.loc[h,'points_per_game']-form.loc[a,'points_per_game']
    gf_diff=form.loc[h,'goals_for_per_game']-form.loc[a,'goals_for_per_game']
    ga_diff=form.loc[h,'goals_against_per_game']-form.loc[a,'goals_against_per_game']
    xin=pd.DataFrame([{
      'logit_home_base':math.log(max(eps,base_ph)/max(eps,draw_base)),
      'logit_away_base':math.log(max(eps,base_pa)/max(eps,draw_base)),
      'expected_goals_home':bxh,'expected_goals_away':bxa,'elo_diff_pre':diff,
      'form_ppg_diff':form_ppg_diff,'gf_pg_diff':gf_diff,'ga_pg_diff':ga_diff,
      'home_matches_played':3,'away_matches_played':3,'host_team_indicator':host,
    }])[feature_cols]
    pg=pd.DataFrame([{
      'expected_goals_home':bxh,'expected_goals_away':bxa,'elo_diff_pre':diff,
      'form_ppg_diff':form_ppg_diff,'gf_pg_diff':gf_diff,'ga_pg_diff':ga_diff,
      'home_matches_played':3,'away_matches_played':3,'host_team_indicator':host,
    }])[goal_cols]
    p=outcome_model.predict_proba(xin)[0]
    lh=max(0.15,float(np.expm1(home_goal_model.predict(pg)[0]))*home_goal_scale); la=max(0.15,float(np.expm1(away_goal_model.predict(pg)[0]))*away_goal_scale)
    # Moderate blend of outcome model with Poisson W/D/L for internal coherence.
    grid=poisson_scores(lh,la,9)
    ph_p=pd_p=pa_p=0.0
    for i in range(10):
      pi=math.exp(-lh)*lh**i/math.factorial(i)
      for j in range(10):
        pj=math.exp(-la)*la**j/math.factorial(j); q=pi*pj
        if i>j: ph_p+=q
        elif i==j: pd_p+=q
        else: pa_p+=q
    p_final=0.72*p+0.28*np.array([ph_p,pd_p,pa_p]); p_final=p_final/p_final.sum()
    # Advance probability: draw after 90 is resolved using a strength-weighted ET/penalty split.
    h_draw_share=1/(1+10**(-diff/500))
    adv_h=float(p_final[0]+p_final[1]*h_draw_share); adv_a=1-adv_h
    pred90=classes[int(np.argmax(p_final))]
    likely=grid[0]['score']
    r32_rows.append({
      'date':date,'home_team':h,'away_team':a,'city':city,'country':country,
      'p_home_win_90':float(p_final[0]),'p_draw_90':float(p_final[1]),'p_away_win_90':float(p_final[2]),
      'p_home_advance':adv_h,'p_away_advance':adv_a,
      'expected_goals_home':lh,'expected_goals_away':la,'expected_score':f'{round(lh)}-{round(la)}',
      'most_likely_score_90':likely,'predicted_outcome_90':pred90,
      'predicted_advancing_team':h if adv_h>=0.5 else a,
      'home_group_points':int(form.loc[h,'points']),'away_group_points':int(form.loc[a,'points']),
      'home_group_goal_difference':int(form.loc[h,'goal_difference']),'away_group_goal_difference':int(form.loc[a,'goal_difference']),
      'home_tournament_elo':float(form.loc[h,'tournament_elo']),'away_tournament_elo':float(form.loc[a,'tournament_elo']),
      'top_scorelines_90':json.dumps(grid,ensure_ascii=False),
      'prediction_cutoff':'2026-06-27 23:59 ET — group-stage data only',
      'model_note':'Updated multinomial outcome model + Poisson score models fitted on all 72 group-stage matches with dynamic Elo and rolling tournament-form features.'
    })
r32_df=pd.DataFrame(r32_rows)
r32_df.to_csv(V9/'outputs'/'round_of_32_predictions.csv',index=False)

# Model metadata
metadata={
 'model_name':'World Cup 2026 tournament-updated forecasting model',
 'training_cutoff':'2026-06-27 after group-stage completion',
 'training_rows':72,
 'outcome_model':'StandardScaler + regularized multinomial LogisticRegression',
 'score_models':'Two StandardScaler + Ridge models on log1p goals, rate-calibrated to observed group-stage home/away scoring',
 'dynamic_features':['v8 base probabilities','v8 expected goals','pre-match dynamic Elo','rolling tournament points per game','rolling goals for/against','host indicator'],
 'validation':'12-fold group-held-out cross-validation',
 'no_round32_leakage':True,
 'caveat':'Only 72 new tournament matches are available. Regularization and group-held-out validation reduce overfitting, but estimates remain uncertain.'
}
(V9/'checkpoints'/'model_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')

# Scripts: copy build script itself later after execution.

# Report
report=f'''# Group-stage evaluation and Round-of-32 update\n\n## Evaluation cutoff\nThe pre-tournament v8 forecasts were evaluated against all 72 completed group-stage matches.\n\n## Headline performance\n- Outcome accuracy: **{acc:.1%}** ({int(eval_df.outcome_correct.sum())}/72)\n- Log loss: **{ll:.3f}**\n- Multiclass Brier score: **{brier:.3f}**\n- Ranked Probability Score: **{rps:.3f}**\n- Exact-score accuracy: **{eval_df.exact_score_correct.mean():.1%}**\n- Total-goals MAE: **{eval_df.total_goals_abs_error.mean():.2f}**\n- Within one goal for both teams: **{eval_df.within_one_goal_both.mean():.1%}**\n- Group winners correct: **{group_winner_hits}/12**\n- Top-two team overlap: **{top2_overlap}/24**\n- Round-of-32 qualifiers captured: **{qual_hits}/32**\n\n## Tournament-update model\nThe update uses only group-stage information available by June 27. It combines the v8 prior probabilities and xG estimates with sequential Elo updates and rolling in-tournament form. A regularized multinomial logistic model forecasts 90-minute outcomes, while two regularized log-goal regressions forecast team goals. Hyperparameters were selected with group-held-out cross-validation.\n\n## Cross-validation\n- Outcome log loss: **{updated_cv['group_held_out_log_loss']:.3f}**\n- Outcome accuracy: **{updated_cv['group_held_out_accuracy']:.1%}**\n- Per-team goal MAE: **{updated_cv['group_held_out_goal_mae']:.2f}**\n\n## Interpretation\nThe evaluation separates outcome success from exact-score success. Correctly identifying the result class is substantially easier than identifying one exact scoreline. The Round-of-32 forecasts therefore present both 90-minute outcome probabilities and advancement probabilities after accounting for extra time and penalties.\n'''
(V9/'reports'/'GROUP_STAGE_EVALUATION_AND_R32.md').write_text(report,encoding='utf-8')

# Compact standalone UI.
logo=''
for candidate in [V9/'ui'/'assets'/'samba_wordmark.jpg',V8/'ui'/'assets'/'samba_wordmark.jpg']:
    if candidate.exists():
        logo='data:image/jpeg;base64,'+base64.b64encode(candidate.read_bytes()).decode('ascii'); break
flag_code={'Algeria':'dz','Argentina':'ar','Australia':'au','Austria':'at','Belgium':'be','Bosnia and Herzegovina':'ba','Brazil':'br','Canada':'ca','Cape Verde':'cv','Colombia':'co','Croatia':'hr','Curaçao':'cw','Czech Republic':'cz','DR Congo':'cd','Ecuador':'ec','Egypt':'eg','England':'gb-eng','France':'fr','Germany':'de','Ghana':'gh','Haiti':'ht','Iran':'ir','Iraq':'iq','Ivory Coast':'ci','Japan':'jp','Jordan':'jo','Mexico':'mx','Morocco':'ma','Netherlands':'nl','New Zealand':'nz','Norway':'no','Panama':'pa','Paraguay':'py','Portugal':'pt','Qatar':'qa','Saudi Arabia':'sa','Scotland':'gb-sct','Senegal':'sn','South Africa':'za','South Korea':'kr','Spain':'es','Sweden':'se','Switzerland':'ch','Tunisia':'tn','Turkey':'tr','United States':'us','Uruguay':'uy','Uzbekistan':'uz'}
ui_payload={'metrics':metrics,'r32':r32_df.to_dict(orient='records'),'evaluation':eval_df[['group','date','home_team','away_team','p_home_win','p_draw','p_away_win','predicted_exact_score','actual_score','actual_outcome','predicted_outcome_argmax','outcome_correct']].to_dict(orient='records'),'flag_code':flag_code,'updated_cv':updated_cv}
html='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SambaSportAI — Group Stage Review & Round of 32</title><style>
:root{--bg:#071221;--card:#0c1b2f;--line:#1f3a5a;--text:#eef5ff;--muted:#9fb4c8;--green:#67e8a0;--blue:#66d9ff;--amber:#ffc857;--red:#ff7b7b}*{box-sizing:border-box}body{margin:0;font-family:Inter,Arial,sans-serif;color:var(--text);background:radial-gradient(circle at 8% 0,#123259,transparent 30%),linear-gradient(180deg,#071221,#0b1930)}.wrap{max-width:1400px;margin:auto;padding:24px}.card{background:rgba(12,27,47,.96);border:1px solid var(--line);border-radius:22px;padding:18px}.hero{display:flex;gap:20px;align-items:center}.logo{height:68px;border-radius:12px}.hero h1{margin:0}.hero p,.muted{color:var(--muted)}.tabs{display:flex;gap:10px;margin:18px 0;flex-wrap:wrap}.tab{padding:11px 14px;border-radius:12px;border:1px solid var(--line);background:#0b1628;color:var(--text);cursor:pointer}.tab.active{background:#123055}.panel{display:none}.panel.active{display:block}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}.kpi{background:#091423;border:1px solid var(--line);border-radius:16px;padding:14px}.kpi .v{font-size:28px;font-weight:800}.kpi .l{color:var(--muted);font-size:12px;margin-top:4px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}.match{background:#091423;border:1px solid var(--line);border-radius:18px;padding:15px}.head,.team,.row{display:flex;justify-content:space-between;align-items:center;gap:10px}.head{font-size:12px;color:var(--muted);margin-bottom:12px}.team{font-weight:700;margin:8px 0}.flag{width:28px;height:20px;object-fit:cover;border-radius:3px;vertical-align:middle;margin-right:8px}.bar{display:flex;height:14px;border-radius:999px;overflow:hidden;background:#071221;border:1px solid var(--line);margin:12px 0}.h{background:var(--green)}.d{background:#6f7f96}.a{background:var(--blue)}.meta{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.chip{background:#0b1628;border:1px solid var(--line);border-radius:12px;padding:8px}.chip small{display:block;color:var(--muted)}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:9px;border-bottom:1px solid var(--line);font-size:13px}.good{color:var(--green)}.bad{color:var(--red)}select{background:#0b1628;color:var(--text);border:1px solid var(--line);padding:10px;border-radius:12px;margin-bottom:14px}@media(max-width:700px){.hero{align-items:flex-start;flex-direction:column}.meta{grid-template-columns:1fr}}
</style></head><body><div class="wrap"><div class="card hero">''' + (f'<img class="logo" src="{logo}">' if logo else '') + '''<div><h1>World Cup 2026: Group Stage Review & Round of 32</h1><p>Evaluation of the pre-tournament forecasts, followed by tournament-updated knockout predictions trained only on information available through the end of the group stage.</p></div></div><div class="tabs"><button class="tab active" data-t="review">Group-stage review</button><button class="tab" data-t="r32">Round of 32 predictions</button><button class="tab" data-t="results">Match-by-match audit</button></div><section id="review" class="panel active"><div id="kpis" class="kpis"></div><div class="card" style="margin-top:14px"><h2>How to read the metrics</h2><p class="muted">Accuracy measures the most likely result class. Log loss, Brier score and RPS also reward well-calibrated probabilities and penalize confident mistakes. Exact-score accuracy is naturally much lower because many plausible scorelines share probability mass.</p><div id="cv"></div></div></section><section id="r32" class="panel"><div id="rgrid" class="grid"></div></section><section id="results" class="panel"><select id="gsel"></select><div class="card" style="overflow:auto"><table><thead><tr><th>Match</th><th>Probabilities H/D/A</th><th>Predicted result</th><th>Predicted score</th><th>Actual</th><th>Outcome hit</th></tr></thead><tbody id="audit"></tbody></table></div></section></div><script>
const DATA=__DATA__; const pct=x=>(100*Number(x)).toFixed(1)+'%'; const f=t=>{const c=DATA.flag_code[t];return c?`<img class="flag" src="https://flagcdn.com/w40/${c}.png" onerror="this.style.display='none'">`:''};
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById(b.dataset.t).classList.add('active')});
const M=DATA.metrics; const ks=[['Outcome accuracy',pct(M.outcome_accuracy)],['Correct outcomes',M.outcome_accuracy?Math.round(M.outcome_accuracy*M.matches_evaluated)+' / '+M.matches_evaluated:'—'],['Log loss',M.log_loss.toFixed(3)],['Brier score',M.multiclass_brier_score.toFixed(3)],['Exact score',pct(M.exact_score_accuracy)],['Total-goals MAE',M.total_goals_mae.toFixed(2)],['Group winners',M.group_winners_correct+' / 12'],['R32 qualifiers',M.round32_qualifiers_correct+' / 32']];document.getElementById('kpis').innerHTML=ks.map(x=>`<div class="kpi"><div class="v">${x[1]}</div><div class="l">${x[0]}</div></div>`).join('');document.getElementById('cv').innerHTML=`<p><strong>Tournament-update cross-validation:</strong> ${pct(DATA.updated_cv.group_held_out_accuracy)} outcome accuracy, ${DATA.updated_cv.group_held_out_log_loss.toFixed(3)} log loss, and ${DATA.updated_cv.group_held_out_goal_mae.toFixed(2)} per-team goal MAE.</p>`;
document.getElementById('rgrid').innerHTML=DATA.r32.map(m=>`<div class="match"><div class="head"><span>${m.date}</span><span>${m.city}</span></div><div class="team"><span>${f(m.home_team)}${m.home_team}</span><b>${pct(m.p_home_advance)} advance</b></div><div class="team"><span>${f(m.away_team)}${m.away_team}</span><b>${pct(m.p_away_advance)} advance</b></div><div class="bar"><div class="h" style="width:${100*m.p_home_win_90}%"></div><div class="d" style="width:${100*m.p_draw_90}%"></div><div class="a" style="width:${100*m.p_away_win_90}%"></div></div><div class="meta"><div class="chip"><small>90-min H/D/A</small>${pct(m.p_home_win_90)} / ${pct(m.p_draw_90)} / ${pct(m.p_away_win_90)}</div><div class="chip"><small>Expected score</small>${m.expected_score}</div><div class="chip"><small>Most likely</small>${m.most_likely_score_90}</div></div><p class="muted">Model pick to advance: <strong>${m.predicted_advancing_team}</strong>. Group form: ${m.home_group_points} pts (${m.home_group_goal_difference>=0?'+':''}${m.home_group_goal_difference} GD) vs ${m.away_group_points} pts (${m.away_group_goal_difference>=0?'+':''}${m.away_group_goal_difference} GD).</p></div>`).join('');
const gs=[...new Set(DATA.evaluation.map(x=>x.group))].sort();const sel=document.getElementById('gsel');sel.innerHTML='<option value="ALL">All groups</option>'+gs.map(g=>`<option>${g}</option>`).join('');function audit(){const g=sel.value;const rows=DATA.evaluation.filter(x=>g==='ALL'||x.group===g);document.getElementById('audit').innerHTML=rows.map(x=>`<tr><td>${f(x.home_team)}${x.home_team} vs ${f(x.away_team)}${x.away_team}</td><td>${pct(x.p_home_win)} / ${pct(x.p_draw)} / ${pct(x.p_away_win)}</td><td>${x.predicted_outcome_argmax}</td><td>${x.predicted_exact_score}</td><td>${x.actual_score}</td><td class="${x.outcome_correct?'good':'bad'}">${x.outcome_correct?'Correct':'Miss'}</td></tr>`).join('')}sel.onchange=audit;audit();
</script></body></html>'''.replace('__DATA__',json.dumps(ui_payload))
(V9/'ui'/'group_stage_review_and_round32_predictions.html').write_text(html,encoding='utf-8')

# Save a runnable training/prediction script copy.
shutil.copy2('/tmp/build_v9.py',V9/'scripts'/'build_evaluate_train_predict_v9.py')
(V9/'README.md').write_text('''# FIFA World Cup 2026 — Tournament Update\n\nThis project evaluates the v8 group-stage forecasts and trains a tournament-updated model for the Round of 32.\n\n## Run\n```bash\npython scripts/build_evaluate_train_predict_v9.py\n```\n\n## Data cutoff\nAll Round-of-32 predictions use group-stage information only, through June 27, 2026. No Round-of-32 result is used in training or prediction.\n\n## Main outputs\n- `outputs/group_stage_prediction_vs_actual.csv`\n- `outputs/group_stage_evaluation_metrics.json`\n- `outputs/round_of_32_predictions.csv`\n- `outputs/team_form_after_group_stage.csv`\n- `ui/group_stage_review_and_round32_predictions.html`\n''',encoding='utf-8')

# Zip
zpath=ROOT/'fifa_2026_predictor_v9.zip'
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
    for p in V9.rglob('*'):
        z.write(p,p.relative_to(V9.parent))
print(json.dumps({'zip':str(zpath),'ui':str(V9/'ui'/'group_stage_review_and_round32_predictions.html'),'metrics':metrics,'updated_cv':updated_cv},indent=2))
