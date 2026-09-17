imgs = set(missing.image_name)
existing = df[df.image_name.isin(imgs)]
print('существующих боксов:', len(existing))

cols = [c for c in df.columns if c in missing.columns]
upload_df = pd.concat([existing[cols], missing[cols]], ignore_index=True)
print('всего боксов:', len(upload_df), 'кадров:', upload_df.image_name.nunique())



upload_df['issue_text'] = None
upload_df['issue_state'] = None
mask = upload_df.index >= len(existing)
upload_df.loc[mask, 'issue_text'] = 'Возможный пропуск (модель)'
upload_df.loc[mask, 'issue_state'] = 'new'


upload_df['issue_text'] = None
upload_df['issue_state'] = None
mask = upload_df.index >= len(existing)
upload_df.loc[mask, 'issue_text'] = 'Возможный пропуск (модель)'
upload_df.loc[mask, 'issue_state'] = 'new'



print(set(upload_df.instance_label.dropna()) - set(df.instance_label.dropna()))
