
/*****************************************************************************
** Macro: Score Data Mining Model
**
** Description: Score the specified model using the score execution service
******************************************************************************/

/**********************************************************************************************
 * Edit the following options to customize the scoring operation:
 * projectId:       REQUIRED: The ID of the project that contains the model to score.
 * modelId:         REQUIRED: The ID of the model to be used for scoring.
 * datasourceUri:   REQUIRED: The URI of the datasource being used for scoring.
 * outputCasLib:    REQUIRED: The name of the CAS library where the scoring output table should be written.
 * outputTableName: REQUIRED: The name of the scoring output table to be created.
**********************************************************************************************/
%let projectId = 5b261b0d-ea07-409f-ae9c-fab69c67a6a3;
%let modelId = ce5b01a4-11b2-41bd-b8b1-7e8380d14e26;
%let datasourceUri = %NRSTR(/dataTables/dataSources/cas~fs~cas-shared-default~fs~Public/tables/CUSTOMER_RETENTION_TRAIN);
%let outputCasLib = CASUSER;
%let outputTableName = score1;


%macro DMScoreModel(projectId, modelId, datasourceUri, outputCasLib, outputTableName)/minoperator;
   filename resp TEMP;
   filename headers TEMP;
   filename data TEMP;

   %let scoreModelUrl=&servicesBaseUrl./dataMiningProjectResources/projects/&projectId/models/&modelId/scoreExecutions;

   %let syscc =0;
   proc json out=data pretty;
      write open object;
      write values "dataTableUri" "&datasourceUri";
      write values "outputCasLibName" "&outputCasLib";
      write values "outputTableName" "&outputTableName";
      write close;
   run;
   %if %sysevalf(&syscc > 4) %then %do;
      %put ERROR: PROC JSON set an error status code of "&syscc".;
      %goto exitM;
   %end;

   %global SYS_PROCHTTP_STATUS_CODE SYS_PROCHTTP_STATUS_PHRASE;
   %let SYS_PROCHTTP_STATUS_CODE=;
   %let SYS_PROCHTTP_STATUS_PHRASE=;

   %let syscc =0;
   proc http
      method="POST"
      OAUTH_BEARER=SAS_SERVICES
      url="&scoreModelUrl"
      in=data
      headerout=headers
      out=resp;
      headers
      "Accept"="application/vnd.sas.score.execution+json"
      "Content-Type"="application/vnd.sas.analytics.data.mining.model.score.request+json";
   run;
   %if %sysevalf(&syscc > 4) %then %do;
      %put ERROR: PROC HTTP set an error status code of "&syscc".;
      %goto exitM;
   %end;

   %if ^(&SYS_PROCHTTP_STATUS_CODE in (200 201 202)) %then %do; /* OK, CREATED, ACCEPTED */
      %put ERROR: PROC HTTP returned an HTTP status code of "&SYS_PROCHTTP_STATUS_CODE" - "&SYS_PROCHTTP_STATUS_PHRASE".;
      %echoFile(resp);
      %goto exitM;
   %end;

   filename mapfile TEMP;
   data _null_;
      file mapfile;
      put '{';
      put '  "DATASETS": [';
      put '    {';
      put '      "DSNAME": "links",';
      put '      "TABLEPATH": "/root/links",';
      put '      "VARIABLES": [';
      put '        {';
      put '          "NAME": "rel",';
      put '          "TYPE": "CHARACTER",';
      put '          "PATH": "/root/links/rel",';
      put '          "CURRENT_LENGTH": 63';
      put '        },';
      put '        {';
      put '          "NAME": "href",';
      put '          "TYPE": "CHARACTER",';
      put '          "PATH": "/root/links/href",';
      put '          "CURRENT_LENGTH": 2047';
      put '        },';
      put '        {';
      put '          "NAME": "type",';
      put '          "TYPE": "CHARACTER",';
      put '          "PATH": "/root/links/type",';
      put '          "CURRENT_LENGTH": 255';
      put '        }';
      put '      ]';
      put '    }';
      put '  ]';
      put '}';
   run;


   libname jsonlib JSON fileref=resp map=mapfile;

   %let selfHref =;
   %let selfType =;

   data _null_;
      set jsonlib.links;
         where rel eq 'self';
      call symput("selfHref", href);
      call symput("selfType", type);
   run;
   filename mapfile;
   libname jsonlib;


   libname jsonlib JSON fileref=resp;

   %let scoreExecutionState=;

   data _null_;
      set jsonlib.root;
      call symput("scoreExecutionState", state);
   run;
   filename resp;
   libname jsonlib;

   /*
   * If the job hasn't completed yet then start polling the job till it is
   * done or we time out.
   */
   %if "&scoreExecutionState" in ("running" "pending") %then %do;
      %determineJobStatus(&selfHref, &selfType, scoreExecutionState);
      %if %sysevalf(&syscc > 4) %then %do;
         %put ERROR: An error was encountered while determining the job status. SYSCC was set to "&syscc".;
         %goto exitM;
      %end;
   %end;

   %if "&scoreExecutionState" eq "completed" %then %goto exitM;
   %else
   %if "&scoreExecutionState" in ("running" "pending") %then %do;
      %put WARNING: The job did not complete yet. It has a current state of "&scoreExecutionState".;
      %goto exitM;
   %end;
   %else %if "&scoreExecutionState" in ("canceled" "timedOut") %then %do;
      %put WARNING: The job has a current state of "&scoreExecutionState".;
      %goto exitM;
   %end;
   %else %do;
      %put ERROR: The job failed. View the log for the job via the jobs icon in SAS Environment Manager. The name of the job will begin with the words 'The Scoring operation for Data Mining model ...'.;
      %goto exitM;
   %end;

   %exitM:
%mend;

%macro echoFile(_fileRef);
   data _null_;
      infile &_fileRef;
      input;
      put _infile_;
   run;
%mend;

%macro determineJobStatus(_jobUri, _mediaType, _rtnVal)/minoperator;
   %let &_rtnVal=running;

   /* Poll for a max of aprox. 100 seconds */
   %let maxPollingAttempts =100;
   %let seconds            =1;
   %let loopLimit          =20;
   %let i                  =0;

   %let operation =JobStatus;

   %let url =&servicesBaseUrl&_jobUri;


   filename resp TEMP;
   %do %while (&i < &maxPollingAttempts);
      %do j=1 %to &loopLimit;
         %let i = %eval(&i + &seconds);
         %if &i >= &maxPollingAttempts %then %goto exitM;
         data _null_;
            call sleep(&seconds, 1);
         run;

         proc http
            out=resp
            method="GET"
            OAUTH_BEARER=SAS_SERVICES
            url="&url"
            ;
            headers
               "Accept"="&_mediaType.+json"
            ;
         run;
         %if %sysevalf(&syscc > 4) %then %do;
            %put ERROR: PROC HTTP set an error status code of "&syscc".;
            %goto exitM;
         %end;

         %if &SYS_PROCHTTP_STATUS_CODE eq 200 %then %do;
            libname job JSON fileref=resp;
            data _null_;
               set job.root;
               call symput("&_rtnVal", trim(state));
            run;
            libname job;
            %if ^("&&&_rtnVal" in ("running" "pending")) %then %goto exitM;
         %end;
         %else %do;
            %let syscc=&SYS_PROCHTTP_STATUS_CODE;
            %put ERROR: An HTTP STATUS code of "&SYS_PROCHTTP_STATUS_CODE" was returned.;
            %goto exitM;
         %end;
      %end;
      %let seconds = %eval (&seconds + 1);
   %end;

   %exitM:
   filename resp;
%mend;


%let servicesBaseUrl =;
data _null_;
   length string $ 1024;
   string= getoption('SERVICESBASEURL');
   call symput('servicesBaseUrl', trim(string));
run;

%DMScoreModel(&projectId, &modelId, &datasourceUri, &outputCasLib, &outputTableName);